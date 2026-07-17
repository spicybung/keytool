bl_info = {
    "name": "Keytool",
    "author": "spicybung",
    "version": (0, 3, 0),
    "blender": (3, 4, 0),
    "location": "File > Import > ReBoot PS1 Texel Object (.TOM)",
    "description": "Imports segmented ReBoot PlayStation 1 TOM models",
    "category": "Import-Export",
}


import argparse
import json
import math
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


try:
    import bpy
    from bpy.props import (
        BoolProperty,
        CollectionProperty,
        EnumProperty,
        FloatProperty,
        StringProperty,
    )
    from bpy.types import Operator, OperatorFileListElement
    from bpy_extras.io_utils import ImportHelper
    from mathutils import Vector
except ImportError:
    bpy = None
    Operator = object
    OperatorFileListElement = object
    ImportHelper = object
    Vector = None


HEADER_SIZE = 0x20
SECTION_TABLE_OFFSET = 0x40
SECTION_COUNT = 9
PACKET_GROUP_COUNTS_OFFSET = 0x64

VERTEX_STRIDE = 0x08
PACKET_STRIDE = 0x1A
COMPACT_TRIANGLE_STRIDE = 0x04
SCRATCH_POINTER_STRIDE = 0x04

MAX_REASONABLE_COUNT = 10_000_000


class TOMParseError(RuntimeError):
    pass


@dataclass(frozen=True)
class TOMHeader:
    material_name_offset: int
    material_name_count: int
    vertex_offset: int
    vertex_count: int
    packet_offset: int
    packet_count: int
    bone_offset: int
    bone_count: int


@dataclass(frozen=True)
class TOMVertex:
    x: int
    y: int
    z: int
    packed_w: int
    source_index: int
    owner_bone_index: int
    scratch_slot: int
    scratch_tag: int


@dataclass(frozen=True)
class TOMPacket:
    primitive: int
    surface_field: int
    packet_vertex_a: int
    packet_vertex_b: int
    packet_vertex_c: int
    uv_a: Tuple[int, int]
    uv_b: Tuple[int, int]
    uv_c: Tuple[int, int]


@dataclass
class TOMBone:
    index: int
    name: str
    local_position: Tuple[int, int, int]
    rigid_vertex_count: int
    source_vertex_start: int
    destination_vertex_start: int
    first_child: int
    next_sibling: int
    runtime_pointer: int
    unresolved_field: int
    parent_index: int = -1


@dataclass
class TOMModel:
    path: Path
    header: TOMHeader
    section_offsets: Tuple[int, ...]
    packet_group_counts: Tuple[int, ...]
    material_names: List[str]
    bone_stride: int
    bone_name_offset: int
    processing_order: List[int]
    scratch_address_base: int
    vertices: List[TOMVertex]
    packets: List[TOMPacket]
    bones: List[TOMBone]
    faces: List[Tuple[int, int, int]]
    warnings: List[str] = field(default_factory=list)


class BinaryReader:
    def __init__(self, data: bytes, source_name: str):
        self.data = data
        self.source_name = source_name
        self.size = len(data)

    def require_range(self, offset: int, size: int, label: str) -> None:
        if offset < 0 or size < 0 or offset + size > self.size:
            raise TOMParseError(
                f"{self.source_name}: {label} is outside the file "
                f"(offset=0x{offset:X}, size=0x{size:X}, file=0x{self.size:X})."
            )

    def unpack_from(self, fmt: str, offset: int, label: str):
        size = struct.calcsize(fmt)
        self.require_range(offset, size, label)
        return struct.unpack_from(fmt, self.data, offset)

    def u8(self, offset: int, label: str) -> int:
        return self.unpack_from("<B", offset, label)[0]

    def u16(self, offset: int, label: str) -> int:
        return self.unpack_from("<H", offset, label)[0]

    def i16(self, offset: int, label: str) -> int:
        return self.unpack_from("<h", offset, label)[0]

    def u32(self, offset: int, label: str) -> int:
        return self.unpack_from("<I", offset, label)[0]

    def i32(self, offset: int, label: str) -> int:
        return self.unpack_from("<i", offset, label)[0]

    def bytes_at(self, offset: int, size: int, label: str) -> bytes:
        self.require_range(offset, size, label)
        return self.data[offset:offset + size]


class TOMParser:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.data = self.path.read_bytes()
        self.reader = BinaryReader(self.data, self.path.name)
        self.warnings: List[str] = []

    def parse(self) -> TOMModel:
        header = self.parse_header()
        section_offsets = self.parse_section_offsets()
        packet_group_counts = self.parse_packet_group_counts(header)
        material_names = self.parse_material_names(header)

        bone_stride = self.detect_bone_stride(
            header=header,
            section_offsets=section_offsets,
        )
        bone_name_offset = self.detect_bone_name_offset(
            header=header,
            bone_stride=bone_stride,
        )

        processing_order = self.parse_processing_order(
            header=header,
            section_offsets=section_offsets,
            bone_stride=bone_stride,
        )

        bones = self.parse_bones(
            header=header,
            bone_stride=bone_stride,
            bone_name_offset=bone_name_offset,
            processing_order=processing_order,
        )
        self.build_bone_parents(bones)

        packets = self.parse_packets(header)
        compact_faces = self.parse_compact_triangle_slots(
            header=header,
            section_offsets=section_offsets,
        )

        scratch_slots, scratch_tags, scratch_address_base = self.parse_scratch_pointers(
            header=header,
            section_offsets=section_offsets,
            compact_faces=compact_faces,
        )

        owner_by_vertex = self.build_vertex_owners(
            header=header,
            bones=bones,
        )

        vertices = self.parse_vertices(
            header=header,
            owner_by_vertex=owner_by_vertex,
            scratch_slots=scratch_slots,
            scratch_tags=scratch_tags,
        )

        faces = self.resolve_faces(
            header=header,
            compact_faces=compact_faces,
            packet_group_counts=packet_group_counts,
            processing_order=processing_order,
            bones=bones,
            scratch_slots=scratch_slots,
        )

        return TOMModel(
            path=self.path,
            header=header,
            section_offsets=section_offsets,
            packet_group_counts=packet_group_counts,
            material_names=material_names,
            bone_stride=bone_stride,
            bone_name_offset=bone_name_offset,
            processing_order=processing_order,
            scratch_address_base=scratch_address_base,
            vertices=vertices,
            packets=packets,
            bones=bones,
            faces=faces,
            warnings=self.warnings.copy(),
        )

    def parse_header(self) -> TOMHeader:
        self.reader.require_range(0, HEADER_SIZE, "TOM header")

        fields = self.reader.unpack_from(
            "<8I",
            0,
            "TOM header fields",
        )

        header = TOMHeader(*fields)

        counts = (
            ("material name count", header.material_name_count),
            ("vertex count", header.vertex_count),
            ("packet count", header.packet_count),
            ("bone count", header.bone_count),
        )

        for label, count in counts:
            if count > MAX_REASONABLE_COUNT:
                raise TOMParseError(
                    f"{self.path.name}: unreasonable {label}: {count}."
                )

        if header.bone_count == 0:
            raise TOMParseError(
                f"{self.path.name}: the model has no bone records."
            )

        self.reader.require_range(
            header.material_name_offset,
            header.material_name_count * 4,
            "material-name table",
        )
        self.reader.require_range(
            header.vertex_offset,
            header.vertex_count * VERTEX_STRIDE,
            "vertex table",
        )
        self.reader.require_range(
            header.packet_offset,
            header.packet_count * PACKET_STRIDE,
            "render-packet table",
        )

        return header

    def parse_section_offsets(self) -> Tuple[int, ...]:
        values = self.reader.unpack_from(
            f"<{SECTION_COUNT}I",
            SECTION_TABLE_OFFSET,
            "TOM trailing-section offset table",
        )

        for section_index, offset in enumerate(values):
            if offset >= self.reader.size:
                raise TOMParseError(
                    f"{self.path.name}: trailing section {section_index} "
                    f"starts outside the file at 0x{offset:X}."
                )

        return tuple(values)

    def parse_packet_group_counts(
        self,
        header: TOMHeader,
    ) -> Tuple[int, ...]:
        counts = self.reader.unpack_from(
            f"<{header.bone_count}I",
            PACKET_GROUP_COUNTS_OFFSET,
            "packet-group count table",
        )

        if sum(counts) != header.packet_count:
            raise TOMParseError(
                f"{self.path.name}: packet-group counts total {sum(counts)}, "
                f"but the header reports {header.packet_count} packets."
            )

        return tuple(counts)

    def parse_material_names(
        self,
        header: TOMHeader,
    ) -> List[str]:
        names: List[str] = []

        for material_index in range(header.material_name_count):
            offset = header.material_name_offset + material_index * 4
            raw_name = self.reader.bytes_at(
                offset,
                4,
                f"material name {material_index}",
            )
            name = raw_name.decode("latin1", errors="replace").rstrip("\0 ")
            names.append(name or f"material_{material_index:02d}")

        return names

    def detect_bone_stride(
        self,
        header: TOMHeader,
        section_offsets: Tuple[int, ...],
    ) -> int:
        pointer_count = header.bone_count - 1

        if pointer_count <= 0:
            section_gap = section_offsets[0] - header.bone_offset
            if section_gap <= 0:
                raise TOMParseError(
                    f"{self.path.name}: cannot derive the single-bone record size."
                )
            return section_gap

        pointer_table_offset = section_offsets[8]
        pointers = self.reader.unpack_from(
            f"<{pointer_count}I",
            pointer_table_offset,
            "runtime bone-pointer table",
        )

        pointer_base = min(pointers)
        pointer_differences = [
            abs(pointer - pointer_base)
            for pointer in pointers
            if pointer != pointer_base
        ]

        derived_stride = 0

        for difference in pointer_differences:
            derived_stride = math.gcd(derived_stride, difference)

        candidates: List[int] = []

        if 0x20 <= derived_stride <= 0x200 and derived_stride % 4 == 0:
            candidates.append(derived_stride)

        for known_stride in (0x40, 0x58):
            if known_stride not in candidates:
                candidates.append(known_stride)

        best_stride = 0
        best_score = -1

        for candidate in candidates:
            score = self.score_bone_stride(
                header=header,
                pointers=pointers,
                stride=candidate,
            )

            if score > best_score:
                best_score = score
                best_stride = candidate

        if best_score < header.bone_count * 3:
            raise TOMParseError(
                f"{self.path.name}: could not determine a credible bone stride."
            )

        return best_stride

    def score_bone_stride(
        self,
        header: TOMHeader,
        pointers: Sequence[int],
        stride: int,
    ) -> int:
        table_size = header.bone_count * stride

        if header.bone_offset + table_size > self.reader.size:
            return -1

        pointer_base = min(pointers)
        pointer_indices = []
        score = 0

        for pointer in pointers:
            difference = pointer - pointer_base

            if difference < 0 or difference % stride != 0:
                return -1

            bone_index = difference // stride + 1

            if not 1 <= bone_index < header.bone_count:
                return -1

            pointer_indices.append(bone_index)
            score += 3

        if len(set(pointer_indices)) == len(pointer_indices):
            score += header.bone_count

        for bone_index in range(header.bone_count):
            record_offset = header.bone_offset + bone_index * stride

            rigid_vertex_count = self.reader.u16(
                record_offset + 0x08,
                f"bone {bone_index} rigid vertex count",
            )
            source_vertex_start = self.reader.u16(
                record_offset + 0x0C,
                f"bone {bone_index} source vertex start",
            )
            first_child = self.reader.i32(
                record_offset + 0x10,
                f"bone {bone_index} first child",
            )
            next_sibling = self.reader.i32(
                record_offset + 0x14,
                f"bone {bone_index} next sibling",
            )

            if rigid_vertex_count == 0:
                score += 1
            elif (
                source_vertex_start < header.vertex_count
                and source_vertex_start + rigid_vertex_count <= header.vertex_count
            ):
                score += 3

            if -1 <= first_child < header.bone_count:
                score += 1

            if -1 <= next_sibling < header.bone_count:
                score += 1

        return score

    def detect_bone_name_offset(
        self,
        header: TOMHeader,
        bone_stride: int,
    ) -> int:
        best_offset = -1
        best_score = -1

        for candidate_offset in range(0x20, bone_stride, 4):
            score = 0
            maximum_length = min(32, bone_stride - candidate_offset)

            if maximum_length <= 0:
                continue

            for bone_index in range(header.bone_count):
                record_offset = (
                    header.bone_offset
                    + bone_index * bone_stride
                    + candidate_offset
                )
                raw_name = self.reader.bytes_at(
                    record_offset,
                    maximum_length,
                    f"candidate bone name {bone_index}",
                )
                name_bytes = raw_name.split(b"\0", 1)[0]

                if not name_bytes:
                    continue

                if all(32 <= byte < 127 for byte in name_bytes):
                    score += 10 + len(name_bytes)

            if score > best_score:
                best_score = score
                best_offset = candidate_offset

        if best_offset < 0 or best_score <= 0:
            raise TOMParseError(
                f"{self.path.name}: could not locate the bone-name field."
            )

        return best_offset

    def parse_processing_order(
        self,
        header: TOMHeader,
        section_offsets: Tuple[int, ...],
        bone_stride: int,
    ) -> List[int]:
        pointer_count = header.bone_count - 1

        if pointer_count <= 0:
            return [0]

        pointers = self.reader.unpack_from(
            f"<{pointer_count}I",
            section_offsets[8],
            "runtime bone-pointer table",
        )
        pointer_base = min(pointers)

        order: List[int] = []

        for pointer_index, pointer in enumerate(pointers):
            difference = pointer - pointer_base

            if difference < 0 or difference % bone_stride != 0:
                raise TOMParseError(
                    f"{self.path.name}: runtime bone pointer {pointer_index} "
                    f"does not align to the detected 0x{bone_stride:X}-byte stride."
                )

            bone_index = difference // bone_stride + 1

            if not 1 <= bone_index < header.bone_count:
                raise TOMParseError(
                    f"{self.path.name}: runtime bone pointer {pointer_index} "
                    f"resolved to invalid bone index {bone_index}."
                )

            order.append(bone_index)

        if len(set(order)) != len(order):
            raise TOMParseError(
                f"{self.path.name}: runtime bone-pointer table contains duplicates."
            )

        if set(order) != set(range(1, header.bone_count)):
            raise TOMParseError(
                f"{self.path.name}: runtime bone-pointer table does not cover "
                "all non-root bones."
            )

        order.append(0)
        return order

    def parse_bones(
        self,
        header: TOMHeader,
        bone_stride: int,
        bone_name_offset: int,
        processing_order: Sequence[int],
    ) -> List[TOMBone]:
        runtime_pointer_by_bone = [0] * header.bone_count

        if header.bone_count > 1:
            pointer_values = self.reader.unpack_from(
                f"<{header.bone_count - 1}I",
                self.parse_section_offsets()[8],
                "runtime bone-pointer table",
            )

            for order_index, bone_index in enumerate(processing_order[:-1]):
                runtime_pointer_by_bone[bone_index] = pointer_values[order_index]

        bones: List[TOMBone] = []

        for bone_index in range(header.bone_count):
            record_offset = header.bone_offset + bone_index * bone_stride

            local_position = self.reader.unpack_from(
                "<3h",
                record_offset,
                f"bone {bone_index} local position",
            )
            rigid_vertex_count = self.reader.u16(
                record_offset + 0x08,
                f"bone {bone_index} rigid vertex count",
            )
            unresolved_field = self.reader.u16(
                record_offset + 0x0C,
                f"bone {bone_index} unresolved field",
            )
            source_vertex_start = unresolved_field
            first_child = self.reader.i32(
                record_offset + 0x10,
                f"bone {bone_index} first child",
            )
            next_sibling = self.reader.i32(
                record_offset + 0x14,
                f"bone {bone_index} next sibling",
            )
            destination_vertex_start = self.reader.i32(
                record_offset + 0x18,
                f"bone {bone_index} destination vertex start",
            )

            maximum_name_length = min(
                32,
                bone_stride - bone_name_offset,
            )
            raw_name = self.reader.bytes_at(
                record_offset + bone_name_offset,
                maximum_name_length,
                f"bone {bone_index} name",
            )
            name = raw_name.split(b"\0", 1)[0].decode(
                "latin1",
                errors="replace",
            )

            if not name:
                name = f"bone_{bone_index:02d}"

            if rigid_vertex_count == 0:
                source_vertex_start = -1
            elif (
                source_vertex_start < 0
                or source_vertex_start + rigid_vertex_count > header.vertex_count
            ):
                raise TOMParseError(
                    f"{self.path.name}: bone {bone_index} ({name}) owns an invalid "
                    f"source range {source_vertex_start}:"
                    f"{source_vertex_start + rigid_vertex_count}."
                )

            bones.append(
                TOMBone(
                    index=bone_index,
                    name=name,
                    local_position=tuple(local_position),
                    rigid_vertex_count=rigid_vertex_count,
                    source_vertex_start=source_vertex_start,
                    destination_vertex_start=destination_vertex_start,
                    first_child=first_child,
                    next_sibling=next_sibling,
                    runtime_pointer=runtime_pointer_by_bone[bone_index],
                    unresolved_field=unresolved_field,
                )
            )

        return bones

    def build_bone_parents(self, bones: List[TOMBone]) -> None:
        for parent in bones:
            child_index = parent.first_child
            visited: set[int] = set()

            while child_index >= 0:
                if child_index >= len(bones):
                    raise TOMParseError(
                        f"{self.path.name}: bone {parent.index} references invalid "
                        f"child {child_index}."
                    )

                if child_index in visited:
                    raise TOMParseError(
                        f"{self.path.name}: sibling loop under bone {parent.index}."
                    )

                visited.add(child_index)
                child = bones[child_index]

                if child.parent_index not in (-1, parent.index):
                    self.warnings.append(
                        f"Bone {child.index} ({child.name}) is referenced by "
                        f"multiple parents."
                    )

                child.parent_index = parent.index
                child_index = child.next_sibling

    def parse_packets(self, header: TOMHeader) -> List[TOMPacket]:
        packets: List[TOMPacket] = []

        for packet_index in range(header.packet_count):
            offset = header.packet_offset + packet_index * PACKET_STRIDE

            primitive = self.reader.u8(
                offset,
                f"packet {packet_index} primitive",
            )
            surface_field = self.reader.u8(
                offset + 0x01,
                f"packet {packet_index} surface field",
            )
            packet_vertex_a, packet_vertex_b, packet_vertex_c = self.reader.unpack_from(
                "<3H",
                offset + 0x02,
                f"packet {packet_index} legacy vertex fields",
            )

            uv_a = self.reader.unpack_from(
                "<2B",
                offset + 0x0A,
                f"packet {packet_index} UV A",
            )
            uv_b = self.reader.unpack_from(
                "<2B",
                offset + 0x0E,
                f"packet {packet_index} UV B",
            )
            uv_c = self.reader.unpack_from(
                "<2B",
                offset + 0x12,
                f"packet {packet_index} UV C",
            )

            packets.append(
                TOMPacket(
                    primitive=primitive,
                    surface_field=surface_field,
                    packet_vertex_a=packet_vertex_a,
                    packet_vertex_b=packet_vertex_b,
                    packet_vertex_c=packet_vertex_c,
                    uv_a=tuple(uv_a),
                    uv_b=tuple(uv_b),
                    uv_c=tuple(uv_c),
                )
            )

        return packets

    def parse_compact_triangle_slots(
        self,
        header: TOMHeader,
        section_offsets: Tuple[int, ...],
    ) -> List[Tuple[int, int, int]]:
        offset = section_offsets[0]
        self.reader.require_range(
            offset,
            header.packet_count * COMPACT_TRIANGLE_STRIDE,
            "compact triangle table",
        )

        triangles: List[Tuple[int, int, int]] = []

        for packet_index in range(header.packet_count):
            triangle_offset = offset + packet_index * COMPACT_TRIANGLE_STRIDE
            a, b, c, terminator = self.reader.unpack_from(
                "<4B",
                triangle_offset,
                f"compact triangle {packet_index}",
            )

            if terminator != 0:
                self.warnings.append(
                    f"Compact triangle {packet_index} has non-zero byte 3: "
                    f"0x{terminator:02X}."
                )

            triangles.append((a, b, c))

        return triangles

    def parse_scratch_pointers(
        self,
        header: TOMHeader,
        section_offsets: Tuple[int, ...],
        compact_faces: Sequence[Tuple[int, int, int]],
    ) -> Tuple[List[int], List[int], int]:
        table_offset = section_offsets[7]
        self.reader.require_range(
            table_offset,
            header.vertex_count * SCRATCH_POINTER_STRIDE,
            "scratch-pointer table",
        )

        raw_words = list(
            self.reader.unpack_from(
                f"<{header.vertex_count}I",
                table_offset,
                "scratch-pointer table",
            )
        )

        low_addresses = [word & 0x00FFFFFF for word in raw_words]
        tags = [(word >> 24) & 0xFF for word in raw_words]

        minimum_compact_slot = min(
            slot
            for face in compact_faces
            for slot in face
        )
        minimum_low_address = min(low_addresses)
        scratch_address_base = (
            minimum_low_address
            - minimum_compact_slot * SCRATCH_POINTER_STRIDE
        )

        slots: List[int] = []

        for vertex_index, low_address in enumerate(low_addresses):
            difference = low_address - scratch_address_base

            if difference < 0 or difference % SCRATCH_POINTER_STRIDE != 0:
                raise TOMParseError(
                    f"{self.path.name}: vertex {vertex_index} has unaligned "
                    f"scratch address 0x{low_address:06X}."
                )

            slot = difference // SCRATCH_POINTER_STRIDE

            if not 0 <= slot <= 0xFF:
                raise TOMParseError(
                    f"{self.path.name}: vertex {vertex_index} resolved to "
                    f"unsupported scratch slot {slot}."
                )

            slots.append(slot)

        return slots, tags, scratch_address_base

    def build_vertex_owners(
        self,
        header: TOMHeader,
        bones: Sequence[TOMBone],
    ) -> List[int]:
        owners = [-1] * header.vertex_count

        for bone in bones:
            if bone.rigid_vertex_count <= 0:
                continue

            start = bone.source_vertex_start
            end = start + bone.rigid_vertex_count

            for vertex_index in range(start, end):
                if owners[vertex_index] != -1:
                    raise TOMParseError(
                        f"{self.path.name}: vertex {vertex_index} belongs to both "
                        f"bone {owners[vertex_index]} and bone {bone.index}."
                    )

                owners[vertex_index] = bone.index

        uncovered = [
            vertex_index
            for vertex_index, owner in enumerate(owners)
            if owner < 0
        ]

        if uncovered:
            self.warnings.append(
                f"{len(uncovered)} vertices are not covered by a rigid bone range."
            )

        return owners

    def parse_vertices(
        self,
        header: TOMHeader,
        owner_by_vertex: Sequence[int],
        scratch_slots: Sequence[int],
        scratch_tags: Sequence[int],
    ) -> List[TOMVertex]:
        vertices: List[TOMVertex] = []

        for vertex_index in range(header.vertex_count):
            offset = header.vertex_offset + vertex_index * VERTEX_STRIDE
            x, y, z, packed_w = self.reader.unpack_from(
                "<4h",
                offset,
                f"vertex {vertex_index}",
            )

            vertices.append(
                TOMVertex(
                    x=x,
                    y=y,
                    z=z,
                    packed_w=packed_w,
                    source_index=vertex_index,
                    owner_bone_index=owner_by_vertex[vertex_index],
                    scratch_slot=scratch_slots[vertex_index],
                    scratch_tag=scratch_tags[vertex_index],
                )
            )

        return vertices

    def resolve_faces(
        self,
        header: TOMHeader,
        compact_faces: Sequence[Tuple[int, int, int]],
        packet_group_counts: Sequence[int],
        processing_order: Sequence[int],
        bones: Sequence[TOMBone],
        scratch_slots: Sequence[int],
    ) -> List[Tuple[int, int, int]]:
        scratch_cache: Dict[int, int] = {}
        faces: List[Tuple[int, int, int]] = []
        packet_index = 0

        for group_index, group_face_count in enumerate(packet_group_counts):
            bone_index = processing_order[group_index]
            bone = bones[bone_index]

            if bone.rigid_vertex_count > 0:
                start = bone.source_vertex_start
                end = start + bone.rigid_vertex_count

                for source_vertex_index in range(start, end):
                    scratch_cache[
                        scratch_slots[source_vertex_index]
                    ] = source_vertex_index

            for local_face_index in range(group_face_count):
                compact_face = compact_faces[
                    packet_index + local_face_index
                ]

                missing_slots = [
                    slot
                    for slot in compact_face
                    if slot not in scratch_cache
                ]

                if missing_slots:
                    raise TOMParseError(
                        f"{self.path.name}: packet "
                        f"{packet_index + local_face_index} references unloaded "
                        f"scratch slot(s) {missing_slots} while processing "
                        f"bone {bone_index} ({bone.name})."
                    )

                faces.append(
                    (
                        scratch_cache[compact_face[0]],
                        scratch_cache[compact_face[1]],
                        scratch_cache[compact_face[2]],
                    )
                )

            packet_index += group_face_count

        if len(faces) != header.packet_count:
            raise TOMParseError(
                f"{self.path.name}: resolved {len(faces)} faces, expected "
                f"{header.packet_count}."
            )

        return faces


def transform_coordinate(
    coordinate: Sequence[float],
    axis_mode: str,
    scale: float,
) -> Tuple[float, float, float]:
    x, y, z = coordinate

    if axis_mode == "NATIVE":
        result = (x, y, z)
    elif axis_mode == "X_Z_NEG_Y":
        result = (x, z, -y)
    elif axis_mode == "X_NEG_Z_Y":
        result = (x, -z, y)
    elif axis_mode == "Z_Y_NEG_X":
        result = (z, y, -x)
    else:
        raise ValueError(f"Unsupported axis mode: {axis_mode}")

    return (
        result[0] * scale,
        result[1] * scale,
        result[2] * scale,
    )


def build_translation_world_positions(
    bones: Sequence[TOMBone],
) -> List[Tuple[float, float, float]]:
    positions: List[Optional[Tuple[float, float, float]]] = [
        None
    ] * len(bones)

    def resolve_position(bone_index: int, active: set[int]):
        existing = positions[bone_index]

        if existing is not None:
            return existing

        if bone_index in active:
            raise TOMParseError(
                f"Bone parent cycle involving bone {bone_index}."
            )

        active.add(bone_index)
        bone = bones[bone_index]
        local_x, local_y, local_z = bone.local_position

        if bone.parent_index < 0:
            position = (
                float(local_x),
                float(local_y),
                float(local_z),
            )
        else:
            parent_position = resolve_position(
                bone.parent_index,
                active,
            )
            position = (
                parent_position[0] + local_x,
                parent_position[1] + local_y,
                parent_position[2] + local_z,
            )

        active.remove(bone_index)
        positions[bone_index] = position
        return position

    for index in range(len(bones)):
        resolve_position(index, set())

    return [
        position
        if position is not None
        else (0.0, 0.0, 0.0)
        for position in positions
    ]


def normalize_vector(
    vector: Sequence[float],
    fallback: Sequence[float],
) -> Tuple[float, float, float]:
    length = math.sqrt(sum(component * component for component in vector))

    if length <= 0.000001:
        vector = fallback
        length = math.sqrt(sum(component * component for component in vector))

    return tuple(component / length for component in vector)


def cross_product(
    left: Sequence[float],
    right: Sequence[float],
) -> Tuple[float, float, float]:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def dot_product(
    left: Sequence[float],
    right: Sequence[float],
) -> float:
    return sum(a * b for a, b in zip(left, right))


def make_basis_from_x_axis(
    x_axis: Sequence[float],
    up_hint: Sequence[float],
) -> Tuple[Tuple[float, float, float], ...]:
    x_axis = normalize_vector(x_axis, (1.0, 0.0, 0.0))
    up_axis = normalize_vector(up_hint, (0.0, 0.0, 1.0))

    if abs(dot_product(x_axis, up_axis)) > 0.92:
        up_axis = (0.0, 1.0, 0.0)

    z_axis = normalize_vector(
        cross_product(x_axis, up_axis),
        (0.0, 1.0, 0.0),
    )
    y_axis = normalize_vector(
        cross_product(z_axis, x_axis),
        (0.0, 0.0, 1.0),
    )

    return (x_axis, y_axis, z_axis)


def transform_by_basis(
    basis: Sequence[Sequence[float]],
    vector: Sequence[float],
) -> Tuple[float, float, float]:
    return (
        basis[0][0] * vector[0]
        + basis[1][0] * vector[1]
        + basis[2][0] * vector[2],
        basis[0][1] * vector[0]
        + basis[1][1] * vector[1]
        + basis[2][1] * vector[2],
        basis[0][2] * vector[0]
        + basis[1][2] * vector[1]
        + basis[2][2] * vector[2],
    )


def bind_axis_for_bone_name(
    name: str,
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    lower_name = name.lower()

    if "uleg" in lower_name or "lleg" in lower_name:
        return (0.0, 0.0, -1.0), (0.0, 1.0, 0.0)

    if "foot" in lower_name:
        return (0.0, -1.0, 0.0), (0.0, 0.0, 1.0)

    if "pelvis" in lower_name:
        return (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)

    vertical_names = ("torso", "neck", "head", "ctr", "point")

    if any(token in lower_name for token in vertical_names):
        return (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)

    arm_names = ("clav", "uarm", "larm", "hand", "fing")

    if any(token in lower_name for token in arm_names):
        if lower_name.startswith("l_"):
            return (-1.0, 0.0, 0.0), (0.0, 0.0, 1.0)

        return (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)

    return (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)


def build_humanoid_bind_pose(
    bones: Sequence[TOMBone],
) -> Tuple[
    List[Tuple[float, float, float]],
    List[Tuple[Tuple[float, float, float], ...]],
]:
    bases = [
        make_basis_from_x_axis(*bind_axis_for_bone_name(bone.name))
        for bone in bones
    ]
    origins: List[Optional[Tuple[float, float, float]]] = [None] * len(bones)

    def resolve_origin(
        bone_index: int,
        active: set[int],
    ) -> Tuple[float, float, float]:
        existing = origins[bone_index]

        if existing is not None:
            return existing

        if bone_index in active:
            raise TOMParseError(
                f"Bone parent cycle involving bone {bone_index}."
            )

        active.add(bone_index)
        bone = bones[bone_index]

        if bone.parent_index < 0:
            origin = (0.0, 0.0, 0.0)
        elif "pelvis" in bone.name.lower():
            parent_origin = resolve_origin(bone.parent_index, active)
            lateral_distance = max(
                abs(float(bone.local_position[1])),
                abs(float(bone.local_position[2])),
                1.0,
            )
            lower_name = bone.name.lower()

            if lower_name.startswith("l_"):
                lateral_sign = -1.0
            elif lower_name.startswith("r_"):
                lateral_sign = 1.0
            else:
                lateral_sign = 0.0

            origin = (
                parent_origin[0] + lateral_sign * lateral_distance,
                parent_origin[1],
                parent_origin[2] + float(bone.local_position[0]),
            )
        else:
            parent_origin = resolve_origin(bone.parent_index, active)
            parent_offset = transform_by_basis(
                bases[bone.parent_index],
                bone.local_position,
            )
            origin = tuple(
                parent_origin[axis] + parent_offset[axis]
                for axis in range(3)
            )

        active.remove(bone_index)
        origins[bone_index] = origin
        return origin

    for bone_index in range(len(bones)):
        resolve_origin(bone_index, set())

    return [origin or (0.0, 0.0, 0.0) for origin in origins], bases


def build_import_faces(
    model: TOMModel,
    face_source: str,
) -> List[Tuple[int, int, int]]:
    if face_source == "SCRATCH_STREAM":
        return list(model.faces)

    if face_source == "PACKET_TABLE":
        return [
            (
                packet.packet_vertex_a,
                packet.packet_vertex_b,
                packet.packet_vertex_c,
            )
            for packet in model.packets
        ]

    raise ValueError(f"Unsupported face source: {face_source}")


def model_has_humanoid_bone_roles(
    bones: Sequence[TOMBone],
) -> bool:
    names = [bone.name.lower() for bone in bones]
    has_body = any(
        "torso" in name or "pelvis" in name
        for name in names
    )
    has_head = any(
        "head" in name or "neck" in name
        for name in names
    )
    has_limb = any(
        token in name
        for name in names
        for token in ("uarm", "larm", "uleg", "lleg", "hand", "foot")
    )
    return has_body and has_head and has_limb


def build_import_vertices(
    model: TOMModel,
    vertex_space: str,
    axis_mode: str,
    scale: float,
) -> List[Tuple[float, float, float]]:
    if vertex_space == "AUTO_RECONSTRUCT":
        if model_has_humanoid_bone_roles(model.bones):
            effective_vertex_space = "HUMANOID_BIND_GUESS"
        else:
            effective_vertex_space = "SOURCE_LOCAL"
    else:
        effective_vertex_space = vertex_space

    bone_positions = build_translation_world_positions(model.bones)
    bind_origins: List[Tuple[float, float, float]] = []
    bind_bases: List[Tuple[Tuple[float, float, float], ...]] = []

    if effective_vertex_space == "HUMANOID_BIND_GUESS":
        bind_origins, bind_bases = build_humanoid_bind_pose(model.bones)
    result: List[Tuple[float, float, float]] = []

    for vertex in model.vertices:
        x = float(vertex.x)
        y = float(vertex.y)
        z = float(vertex.z)

        if (
            effective_vertex_space == "TRANSLATION_HIERARCHY"
            and vertex.owner_bone_index >= 0
        ):
            bone_position = bone_positions[vertex.owner_bone_index]
            x += bone_position[0]
            y += bone_position[1]
            z += bone_position[2]
        elif (
            effective_vertex_space == "HUMANOID_BIND_GUESS"
            and vertex.owner_bone_index >= 0
        ):
            bone_index = vertex.owner_bone_index
            transformed = transform_by_basis(
                bind_bases[bone_index],
                (x, y, z),
            )
            origin = bind_origins[bone_index]
            x = origin[0] + transformed[0]
            y = origin[1] + transformed[1]
            z = origin[2] + transformed[2]

        result.append(
            transform_coordinate(
                (x, y, z),
                axis_mode=axis_mode,
                scale=scale,
            )
        )

    return result


def create_int_attribute(
    mesh,
    name: str,
    domain: str,
    values: Sequence[int],
) -> None:
    attribute = mesh.attributes.get(name)

    if attribute is None:
        attribute = mesh.attributes.new(
            name=name,
            type="INT",
            domain=domain,
        )

    if len(attribute.data) != len(values):
        raise RuntimeError(
            f"Attribute {name} expects {len(attribute.data)} values, "
            f"received {len(values)}."
        )

    for element, value in zip(attribute.data, values):
        element.value = int(value)


def create_surface_materials(
    mesh,
    packets: Sequence[TOMPacket],
) -> Dict[int, int]:
    surface_fields = sorted(
        {packet.surface_field for packet in packets}
    )
    material_slot_by_surface: Dict[int, int] = {}

    for surface_field in surface_fields:
        material_name = f"TOM_Surface_{surface_field:02d}"
        material = bpy.data.materials.get(material_name)

        if material is None:
            material = bpy.data.materials.new(material_name)

        mesh.materials.append(material)
        material_slot_by_surface[surface_field] = len(mesh.materials) - 1

    return material_slot_by_surface


def assign_packet_metadata(
    mesh,
    model: TOMModel,
    reverse_winding: bool,
    invert_v: bool,
) -> None:
    uv_layer = mesh.uv_layers.new(name="TOM_UV")
    material_slot_by_surface = create_surface_materials(
        mesh,
        model.packets,
    )

    packet_indices = []
    surface_fields = []
    primitives = []

    for polygon_index, polygon in enumerate(mesh.polygons):
        packet = model.packets[polygon_index]

        packet_indices.append(polygon_index)
        surface_fields.append(packet.surface_field)
        primitives.append(packet.primitive)

        polygon.material_index = material_slot_by_surface[
            packet.surface_field
        ]

        packet_uvs = [
            packet.uv_a,
            packet.uv_b,
            packet.uv_c,
        ]

        if reverse_winding:
            packet_uvs = [
                packet_uvs[0],
                packet_uvs[2],
                packet_uvs[1],
            ]

        for loop_offset, loop_index in enumerate(polygon.loop_indices):
            u, v = packet_uvs[loop_offset]
            normalized_u = u / 255.0
            normalized_v = v / 255.0

            if invert_v:
                normalized_v = 1.0 - normalized_v

            uv_layer.data[loop_index].uv = (
                normalized_u,
                normalized_v,
            )

    create_int_attribute(
        mesh,
        "tom_packet_index",
        "FACE",
        packet_indices,
    )
    create_int_attribute(
        mesh,
        "tom_surface_field",
        "FACE",
        surface_fields,
    )
    create_int_attribute(
        mesh,
        "tom_primitive",
        "FACE",
        primitives,
    )


def assign_vertex_metadata(
    mesh,
    model: TOMModel,
) -> None:
    create_int_attribute(
        mesh,
        "tom_source_vertex",
        "POINT",
        [vertex.source_index for vertex in model.vertices],
    )
    create_int_attribute(
        mesh,
        "tom_bone_index",
        "POINT",
        [vertex.owner_bone_index for vertex in model.vertices],
    )
    create_int_attribute(
        mesh,
        "tom_scratch_slot",
        "POINT",
        [vertex.scratch_slot for vertex in model.vertices],
    )
    create_int_attribute(
        mesh,
        "tom_scratch_tag",
        "POINT",
        [vertex.scratch_tag for vertex in model.vertices],
    )
    create_int_attribute(
        mesh,
        "tom_packed_w",
        "POINT",
        [vertex.packed_w for vertex in model.vertices],
    )


def create_vertex_groups(
    mesh_object,
    model: TOMModel,
) -> None:
    for bone in model.bones:
        group = mesh_object.vertex_groups.new(
            name=bone.name,
        )

        if bone.rigid_vertex_count <= 0:
            continue

        start = bone.source_vertex_start
        end = start + bone.rigid_vertex_count

        group.add(
            list(range(start, end)),
            1.0,
            "REPLACE",
        )


def create_debug_armature(
    context,
    collection,
    model: TOMModel,
    vertex_space: str,
    axis_mode: str,
    scale: float,
):
    armature_data = bpy.data.armatures.new(
        f"{model.path.stem}_TOM_Armature",
    )
    armature_object = bpy.data.objects.new(
        f"{model.path.stem}_TOM_Armature",
        armature_data,
    )
    collection.objects.link(armature_object)

    context.view_layer.objects.active = armature_object
    armature_object.select_set(True)

    bpy.ops.object.mode_set(mode="EDIT")

    if vertex_space == "AUTO_RECONSTRUCT":
        use_reconstructed_pose = model_has_humanoid_bone_roles(model.bones)
    else:
        use_reconstructed_pose = vertex_space == "HUMANOID_BIND_GUESS"

    if use_reconstructed_pose:
        world_positions, world_bases = build_humanoid_bind_pose(model.bones)
    else:
        world_positions = build_translation_world_positions(model.bones)
        world_bases = []
    edit_bones = []

    for bone in model.bones:
        edit_bone = armature_data.edit_bones.new(
            bone.name,
        )
        head = transform_coordinate(
            world_positions[bone.index],
            axis_mode=axis_mode,
            scale=scale,
        )
        edit_bone.head = head
        edit_bones.append(edit_bone)

    minimum_tail_length = max(abs(scale) * 8.0, 0.01)

    for bone in model.bones:
        edit_bone = edit_bones[bone.index]

        if use_reconstructed_pose:
            tail_direction = transform_by_basis(
                world_bases[bone.index],
                (1.0, 0.0, 0.0),
            )
            transformed_tail_direction = transform_coordinate(
                tail_direction,
                axis_mode=axis_mode,
                scale=minimum_tail_length,
            )
            edit_bone.tail = (
                edit_bone.head
                + Vector(transformed_tail_direction)
            )
            continue

        child_indices = [
            child.index
            for child in model.bones
            if child.parent_index == bone.index
        ]

        if child_indices:
            child_head = edit_bones[child_indices[0]].head.copy()

            if (child_head - edit_bone.head).length > minimum_tail_length:
                edit_bone.tail = child_head
            else:
                edit_bone.tail = (
                    edit_bone.head
                    + Vector((0.0, minimum_tail_length, 0.0))
                )
        elif bone.parent_index >= 0:
            parent_head = edit_bones[bone.parent_index].head
            direction = edit_bone.head - parent_head

            if direction.length <= minimum_tail_length:
                direction = Vector((0.0, minimum_tail_length, 0.0))
            else:
                direction.normalize()
                direction *= minimum_tail_length

            edit_bone.tail = edit_bone.head + direction
        else:
            edit_bone.tail = (
                edit_bone.head
                + Vector((0.0, minimum_tail_length, 0.0))
            )

    for bone in model.bones:
        if bone.parent_index >= 0:
            edit_bones[bone.index].parent = edit_bones[
                bone.parent_index
            ]

    bpy.ops.object.mode_set(mode="OBJECT")
    armature_object.select_set(False)

    return armature_object


def import_tom_into_blender(
    context,
    path: Path,
    vertex_space: str,
    face_source: str,
    axis_mode: str,
    scale: float,
    reverse_winding: bool,
    invert_v: bool,
    create_armature: bool,
):
    model = TOMParser(path).parse()

    collection = bpy.data.collections.new(
        f"{model.path.stem}_TOM",
    )
    context.scene.collection.children.link(collection)

    imported_vertices = build_import_vertices(
        model=model,
        vertex_space=vertex_space,
        axis_mode=axis_mode,
        scale=scale,
    )

    source_faces = build_import_faces(
        model=model,
        face_source=face_source,
    )

    if reverse_winding:
        imported_faces = [
            (face[0], face[2], face[1])
            for face in source_faces
        ]
    else:
        imported_faces = source_faces

    mesh = bpy.data.meshes.new(
        f"{model.path.stem}_TOM_Mesh",
    )
    mesh.from_pydata(
        imported_vertices,
        [],
        imported_faces,
    )
    mesh.update()

    mesh_object = bpy.data.objects.new(
        model.path.stem,
        mesh,
    )
    collection.objects.link(mesh_object)

    assign_packet_metadata(
        mesh=mesh,
        model=model,
        reverse_winding=reverse_winding,
        invert_v=invert_v,
    )
    assign_vertex_metadata(
        mesh=mesh,
        model=model,
    )
    create_vertex_groups(
        mesh_object=mesh_object,
        model=model,
    )

    mesh_object["tom_source_path"] = str(model.path)
    mesh_object["tom_material_names"] = json.dumps(
        model.material_names,
    )
    mesh_object["tom_bone_stride"] = model.bone_stride
    mesh_object["tom_bone_name_offset"] = model.bone_name_offset
    mesh_object["tom_scratch_address_base"] = (
        model.scratch_address_base
    )
    mesh_object["tom_processing_order"] = model.processing_order
    mesh_object["tom_vertex_space"] = vertex_space
    mesh_object["tom_face_source"] = face_source
    mesh_object["tom_axis_mode"] = axis_mode
    mesh_object["tom_warnings"] = json.dumps(model.warnings)

    armature_object = None

    if create_armature:
        armature_object = create_debug_armature(
            context=context,
            collection=collection,
            model=model,
            vertex_space=vertex_space,
            axis_mode=axis_mode,
            scale=scale,
        )

        modifier = mesh_object.modifiers.new(
            name="TOM Armature",
            type="ARMATURE",
        )
        modifier.object = armature_object
        mesh_object.parent = armature_object

    context.view_layer.objects.active = mesh_object
    mesh_object.select_set(True)

    return model, mesh_object, armature_object


if bpy is not None:
    class KEYTOOL_OT_import_tom(Operator, ImportHelper):
        bl_idname = "import_scene.keytool_tom"
        bl_label = "Import ReBoot PS1 TOM"
        bl_options = {"UNDO", "PRESET"}

        filename_ext = ".tom"

        filter_glob: StringProperty(
            default="*.tom;*.TOM",
            options={"HIDDEN"},
        )

        files: CollectionProperty(
            type=OperatorFileListElement,
            options={"HIDDEN", "SKIP_SAVE"},
        )

        directory: StringProperty(
            subtype="DIR_PATH",
            options={"HIDDEN", "SKIP_SAVE"},
        )

        vertex_space: EnumProperty(
            name="Vertex Space",
            description=(
                "SOURCE_LOCAL preserves the exact stored coordinates. "
                "TRANSLATION_HIERARCHY adds accumulated bone translations "
                "without inventing unknown rest rotations"
            ),
            items=(
                (
                    "AUTO_RECONSTRUCT",
                    "Automatic Reconstruction",
                    "Reconstruct recognized segmented character skeletons and preserve source-local coordinates for unknown layouts",
                ),
                (
                    "SOURCE_LOCAL",
                    "Source Local",
                    "Import exact stored rigid-part coordinates",
                ),
                (
                    "TRANSLATION_HIERARCHY",
                    "Translation Hierarchy",
                    "Add accumulated bone translations; rotations remain unresolved",
                ),
                (
                    "HUMANOID_BIND_GUESS",
                    "Humanoid Bind Reconstruction",
                    "Use the decoded hierarchy and generic bone-name roles to separate humanoid rigid sections",
                ),
            ),
            default="AUTO_RECONSTRUCT",
        )

        face_source: EnumProperty(
            name="Face Source",
            description=(
                "Packet Table reproduces the original complete triangle list. "
                "Scratch Stream exposes the separately decoded runtime scratch references"
            ),
            items=(
                (
                    "PACKET_TABLE",
                    "Legacy Packet Fields (Experimental)",
                    "Interpret the three unresolved 16-bit packet fields as direct vertex indices for comparison only",
                ),
                (
                    "SCRATCH_STREAM",
                    "Scratch Stream",
                    "Use triangles reconstructed through the runtime scratch-slot stream",
                ),
            ),
            default="SCRATCH_STREAM",
        )

        axis_mode: EnumProperty(
            name="Axis Conversion",
            items=(
                (
                    "NATIVE",
                    "Native (X, Y, Z)",
                    "Keep the file axes unchanged",
                ),
                (
                    "X_Z_NEG_Y",
                    "X, Z, -Y",
                    "Swap Y/Z and negate the new Z axis",
                ),
                (
                    "X_NEG_Z_Y",
                    "X, -Z, Y",
                    "Swap Y/Z and negate the new Y axis",
                ),
                (
                    "Z_Y_NEG_X",
                    "Z, Y, -X",
                    "Rotate the coordinate basis without model-name assumptions",
                ),
            ),
            default="NATIVE",
        )

        scale: FloatProperty(
            name="Scale",
            description="Uniform scale applied after axis conversion",
            default=0.01,
            min=0.000001,
            soft_max=1.0,
        )

        reverse_winding: BoolProperty(
            name="Reverse Winding",
            description="Reverse triangle winding and matching UV order",
            default=False,
        )

        invert_v: BoolProperty(
            name="Invert UV V",
            description="Convert packet V from top-down to Blender UV space",
            default=True,
        )

        create_armature: BoolProperty(
            name="Create Debug Armature",
            description=(
                "Create the decoded child/sibling hierarchy using the selected "
                "mesh reconstruction basis"
            ),
            default=True,
        )

        def execute(self, context):
            selected_paths: List[Path] = []

            if self.files:
                base_directory = Path(self.directory)

                for selected_file in self.files:
                    selected_paths.append(
                        base_directory / selected_file.name
                    )
            else:
                selected_paths.append(Path(self.filepath))

            failures: List[str] = []
            imported_count = 0

            for path in selected_paths:
                try:
                    model, mesh_object, armature_object = (
                        import_tom_into_blender(
                            context=context,
                            path=path,
                            vertex_space=self.vertex_space,
                            face_source=self.face_source,
                            axis_mode=self.axis_mode,
                            scale=self.scale,
                            reverse_winding=self.reverse_winding,
                            invert_v=self.invert_v,
                            create_armature=self.create_armature,
                        )
                    )
                    imported_count += 1

                    for warning in model.warnings:
                        print(
                            f"[Keytool][{path.name}] Warning: {warning}"
                        )

                    print(
                        f"[Keytool] Imported {path.name}: "
                        f"{model.header.vertex_count} vertices, "
                        f"{len(model.faces)} triangles, "
                        f"{model.header.bone_count} bones, "
                        f"bone stride 0x{model.bone_stride:X}."
                    )
                except Exception as error:
                    failures.append(
                        f"{path.name}: {error}"
                    )
                    print(
                        f"[Keytool] Failed to import {path}: {error}"
                    )

            if failures:
                self.report(
                    {"ERROR"},
                    " | ".join(failures[:3]),
                )

                if imported_count == 0:
                    return {"CANCELLED"}

            self.report(
                {"INFO"},
                f"Imported {imported_count} TOM file(s).",
            )
            return {"FINISHED"}


    def menu_func_import(self, context):
        self.layout.operator(
            KEYTOOL_OT_import_tom.bl_idname,
            text="ReBoot PS1 Texel Object (.TOM)",
        )


    classes = (
        KEYTOOL_OT_import_tom,
    )


    def register():
        for addon_class in classes:
            bpy.utils.register_class(addon_class)

        bpy.types.TOPBAR_MT_file_import.append(
            menu_func_import
        )


    def unregister():
        bpy.types.TOPBAR_MT_file_import.remove(
            menu_func_import
        )

        for addon_class in reversed(classes):
            bpy.utils.unregister_class(addon_class)


else:
    def register():
        raise RuntimeError(
            "Keytool registration requires Blender's bpy module."
        )


    def unregister():
        return None


def inspect_tom(path: Path) -> dict:
    model = TOMParser(path).parse()

    return {
        "path": str(model.path),
        "file_size": model.path.stat().st_size,
        "header": {
            "material_name_offset": model.header.material_name_offset,
            "material_name_count": model.header.material_name_count,
            "vertex_offset": model.header.vertex_offset,
            "vertex_count": model.header.vertex_count,
            "packet_offset": model.header.packet_offset,
            "packet_count": model.header.packet_count,
            "bone_offset": model.header.bone_offset,
            "bone_count": model.header.bone_count,
        },
        "section_offsets": list(model.section_offsets),
        "bone_stride": model.bone_stride,
        "bone_name_offset": model.bone_name_offset,
        "scratch_address_base": model.scratch_address_base,
        "processing_order": model.processing_order,
        "material_names": model.material_names,
        "bones": [
            {
                "index": bone.index,
                "name": bone.name,
                "parent_index": bone.parent_index,
                "first_child": bone.first_child,
                "next_sibling": bone.next_sibling,
                "local_position": list(bone.local_position),
                "rigid_vertex_count": bone.rigid_vertex_count,
                "source_vertex_start": bone.source_vertex_start,
                "destination_vertex_start": bone.destination_vertex_start,
                "runtime_pointer": bone.runtime_pointer,
            }
            for bone in model.bones
        ],
        "resolved_triangle_count": len(model.faces),
        "warnings": model.warnings,
    }


def run_command_line(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect ReBoot PS1 TOM files without Blender. "
            "The same parser is used by the Blender importer."
        )
    )
    parser.add_argument(
        "tom_files",
        nargs="+",
        type=Path,
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="write_json",
        help="Write one JSON report beside each TOM file.",
    )

    arguments = parser.parse_args(argv)
    exit_code = 0

    for tom_path in arguments.tom_files:
        try:
            report = inspect_tom(tom_path)
            rendered = json.dumps(
                report,
                indent=2,
            )

            if arguments.write_json:
                output_path = tom_path.with_suffix(
                    tom_path.suffix + ".json"
                )
                output_path.write_text(
                    rendered,
                    encoding="utf-8",
                )
                print(output_path)
            else:
                print(rendered)
        except Exception as error:
            print(
                f"{tom_path}: {error}",
                file=sys.stderr,
            )
            exit_code = 1

    return exit_code


if __name__ == "__main__":
    if bpy is not None:
        register()
    else:
        raise SystemExit(
            run_command_line(sys.argv[1:])
        )

import struct
import bpy
from bpy.types import Operator
from bpy.props import StringProperty, IntProperty
from bpy_extras.io_utils import ImportHelper

class ImportTOMHeader(Operator, ImportHelper):
    bl_idname = "import_scene.tom"
    bl_label = "Import ReBoot PS1 .TOM"

    filename_ext = ".tom"
    filter_glob: StringProperty(
        default="*.tom",
        options={'HIDDEN'}
    )

    end_offset: IntProperty(
        name="End Offset",
        description="Manual end offset for vertex/index data block",
        default=0x1000,
        min=0
    )

    def execute(self, context):
        with open(self.filepath, "rb") as f:
            data = f.read()

        if len(data) < 32:
            self.report({'ERROR'}, "File too small for a .TOM header.")
            return {'CANCELLED'}

        fields = struct.unpack("<IIIIIIII", data[:32])

        header = {
            "material_list_offset":  fields[0],
            "num_materials":         fields[1],
            "unknown_offset_1":      fields[2],
            "unknown_offset_2":      fields[3],
            "geometry_data_offset":  fields[4],
            "unknown_offset_3":      fields[5],
            "bone_data_offset":      fields[6],
            "bone_count":            fields[7],
        }

        geo_start = header["geometry_data_offset"]
        geo_end = self.end_offset

        print("== .TOM Header ==")
        for k, v in header.items():
            print(f"{k:<24}: 0x{v:08X} ({v})")

        print("\n== Geometry Block ==")
        print(f"Start Offset        : 0x{geo_start:08X}")
        print(f"User End Offset     : 0x{geo_end:08X} ({geo_end})")
        print(f"Block Size          : {geo_end - geo_start} bytes")

        return {'FINISHED'}


def menu_func_import(self, context):
    self.layout.operator(ImportTOMHeader.bl_idname, text="Import Texel Object (.TOM)")

def register():
    bpy.utils.register_class(ImportTOMHeader)
    bpy.types.TOPBAR_MT_file_import.append(menu_func_import)

def unregister():
    bpy.utils.unregister_class(ImportTOMHeader)
    bpy.types.TOPBAR_MT_file_import.remove(menu_func_import)

if __name__ == "__main__":
    register()
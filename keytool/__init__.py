bl_info = {
    "name": "Keytool",
    "author": "spicybung",
    "version": (0, 3, 0),
    "blender": (3, 4, 0),
    "location": "File > Import > ReBoot PS1 Texel Object (.TOM)",
    "description": "Imports segmented ReBoot PlayStation 1 TOM models",
    "category": "Import-Export",
}

from .parse import register, unregister

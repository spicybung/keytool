bl_info = {
    "name": "Keytool",
    "author": "spicybung",
    "version": (0, 2, 0),
    "blender": (3, 4, 0),
    "location": "File > Import > ReBoot PS1 Texel Object (.TOM)",
    "description": "Imports ReBoot PlayStation 1 TOM models and preserves reverse-engineering metadata",
    "category": "Import-Export",
}

from .parse import register, unregister

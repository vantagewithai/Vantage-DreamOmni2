from .nodes_dreamomni2 import TextEncodeDreamOmni2, VantageAdaptiveImageGrid

NODE_CLASS_MAPPINGS = {
    "TextEncodeDreamOmni2": TextEncodeDreamOmni2,
    "VantageAdaptiveImageGrid": VantageAdaptiveImageGrid,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TextEncodeDreamOmni2": "DreamOmni2 Text Encoder",
    "VantageAdaptiveImageGrid": "Adaptive Image Stitch - Vantage",
}


__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

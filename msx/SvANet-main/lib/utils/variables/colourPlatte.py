try:
    import imgviz
except ImportError:  # imgviz is visualization-only; model construction does not need it.
    imgviz = None


def _label_colormap(size):
    if imgviz is not None:
        return imgviz.label_colormap(size)
    # Pascal-VOC style fallback used only for visualization metadata.
    colormap = []
    for index in range(size):
        red = green = blue = 0
        value = index
        for shift in range(8):
            red |= ((value >> 0) & 1) << (7 - shift)
            green |= ((value >> 1) & 1) << (7 - shift)
            blue |= ((value >> 2) & 1) << (7 - shift)
            value >>= 3
        colormap.append([red, green, blue])
    return colormap


COLOUR_CODES = {
    "twoclasses": [
        [0, 0, 0], # background
        [255, 255, 255]
    ],
    **dict.fromkeys(
        [
            "default", 
            "spermhealth", "atlas", "fives", "kits23"
        ], 
        list(map(list, _label_colormap(256)))
    ),
    }


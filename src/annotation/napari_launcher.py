# launches a napari window and loads the images and the predictions
import torch
import napari
from matplotlib.colors import rgb2hex

from src.utils.annotation_utils import *
from src.utils.utils import *
from src.utils.const import *

def napari_launcher(image, bboxes_xyxy, title):
    """Launches a napari window and loads the images and the predictions."""
    viewer = napari.Viewer(
        title=title,
    )

    # load the image
    # image = skimage.io.imread(image_path, as_gray=True)
    image_layer = viewer.add_image(image)

    # load the prediction
    bboxes_napari = bbox_xyxy_to_napari(bboxes_xyxy)
    if "color" in bboxes_xyxy.columns:
            edge_color = bboxes_xyxy["color"].to_list()
    else:
        edge_color = "blue"
    shapes_layer = viewer.add_shapes(
        data=bboxes_napari,
        face_color='transparent',
        edge_color=edge_color,
        edge_width=4,
        name="organoid detection"
    )
    shapes_layer.current_edge_color = "magenta"
    annotated_bboxes = pd.DataFrame(columns=["x1", "y1", "x2", "y2", "color"])
    
    # hide the boxes
    @viewer.bind_key('h')# hide the boxes
    def hide_boxes(viewer: napari.Viewer):
        shapes_layer.visible = not shapes_layer.visible
        
    # save results
    @viewer.bind_key('s')# when finished
    def save_results(viewer: napari.Viewer):
        dialog = NextDialog()
        if not dialog.exec():
            return
        napari_bboxes = viewer.layers["organoid detection"].data
        colors = viewer.layers["organoid detection"].edge_color
        colors_hex = [rgb2hex(color) for color in colors]
        
        # write the bounding boxes
        corrected_bboxes = bbox_napari_to_xyxy(napari_bboxes=napari_bboxes)
        corrected_bboxes = clip_xyxy_to_image(bboxes=corrected_bboxes, image_shape=image.shape)
        annotated_bboxes[["x1", "y1", "x2", "y2"]] = corrected_bboxes
        annotated_bboxes["color"] = colors_hex

        viewer.close()
    
    @viewer.bind_key('e')
    def stop_annotating(viewer: napari.Viewer):
        dialog = CloseDialog()
        if dialog.exec():
            exit(1)


    # run napari
    napari.run()
    
    return annotated_bboxes
const tf = require('@tensorflow/tfjs');

function convert_to_square(boxes){
    // """Convert bounding boxes to a square form.

    // Parameters:
    //     bboxes: float tensor of shape [n, 4]

    // Returns:
    //     float tensor of shape [n, 4]
    // """

    const x1 = tf.reshape(tf.slice(boxes, [0,0], [-1,1]), [-1])
    const y1 = tf.reshape(tf.slice(boxes, [0,1], [-1,1]), [-1])
    const x2 = tf.reshape(tf.slice(boxes, [0,2], [-1,1]), [-1])
    const y2 = tf.reshape(tf.slice(boxes, [0,3], [-1,1]), [-1])

    const h = tf.sub(y2, y1)
    const w = tf.sub(x2, x1)
    max_side = tf.maximum(h, w)

    const dx1 = tf.sub(tf.add(x1, tf.mul(w, 0.5)), tf.mul(max_side, 0.5))
    const dy1 = tf.sub(tf.add(y1, tf.mul(h, 0.5)), tf.mul(max_side, 0.5))
    const dx2 = tf.add(dx1, max_side)
    const dy2 = tf.add(dy1, max_side)

    return tf.stack([
        tf.round(dx1), 
        tf.round(dy1), 
        tf.round(dx2), 
        tf.round(dy2)
    ], 1)
}

function calibrate_box(boxes, offsets){
    // """Use offsets returned by a network to
    // correct the bounding box coordinates.

    // Parameters:
    //     bboxes: float tensor of shape [n, 4].
    //     offsets: float tensor of shape [n, 4].

    // Returns:
    //     float tensor of shape [n, 4]
    // """

    const x1 = tf.reshape(tf.slice(boxes, [0,0], [-1,1]), [-1])
    const y1 = tf.reshape(tf.slice(boxes, [0,1], [-1,1]), [-1])
    const x2 = tf.reshape(tf.slice(boxes, [0,2], [-1,1]), [-1])
    const y2 = tf.reshape(tf.slice(boxes, [0,3], [-1,1]), [-1])

    const w = tf.sub(x2, x1)
    const h = tf.sub(y2, y1)

    const translation = tf.stack([w, h, w, h], 1).mul(offsets)

    return tf.add(boxes, translation)
}
function preprocess(img){
    // """Preprocess image tensor before applying a network.

    // Parameters:
    //     img: image tensor

    // Returns:
    //     float tensor with shape of img
    // """

    img = tf.mul(tf.sub(img, 127.5), 0.0078125)

    return img
}

function get_image_boxes(boxes, img, height, width, num_boxes, size=24){
    // """Cut out boxes from the image.

    // Parameters:
    //     boxes: a float tensor with shape [num_instance, 4],
    //     img: a tensor with shape [height, width, 3],
    //     height: original height of the image
    //     width: original width of the image
    //     num_boxes: number of boxes to cut out
    //     size: size of the cutouts

    // Returns:
    //     a float tensor with shape [num_boxes, size, size, 3]
    // """

    const x1 = tf.maximum(tf.reshape(tf.slice(boxes, [0,0], [-1,1]), [-1]), 0.0).div(width)
    const y1 = tf.maximum(tf.reshape(tf.slice(boxes, [0,1], [-1,1]), [-1]), 0.0).div(height)
    const x2 = tf.minimum(tf.reshape(tf.slice(boxes, [0,2], [-1,1]), [-1]), width).div(width)
    const y2 = tf.minimum(tf.reshape(tf.slice(boxes, [0,3], [-1,1]), [-1]), height).div(height)

    boxes = tf.stack([y1, x1, y2, x2], 1)
    var img_boxes = tf.image.cropAndResize(tf.expandDims(img, 0), boxes, tf.zeros([num_boxes]).cast('int32'), [size, size])

    img_boxes = preprocess(img_boxes)

    return img_boxes
}

async function generate_boxes(probs, offsets, scale, threshold){
    // """Convert output of PNet to bouding boxes tensor.
    //
    // Parameters:
    //     probs: float tensor of shape [p, m, 2], output of PNet
    //     offsets: float tensor of shape [p, m, 4], output of PNet
    //     scale: float, scale of the input image
    //     threshold: float, confidence threshold

    // Returns:
    //     float tensor of shape [n, 9]
    // """

    const stride = 2
    const cell_size = 12

    probs = tf.slice(probs, [0, 0, 1], [-1, -1, 1])
    probs = tf.squeeze(probs, [2])

    inds = await tf.whereAsync(tf.greater(probs, threshold))

    if (inds.shape[0] == 0) {
        return tf.zeros([0, 9])
    }

    // offsets: N x 4
    offsets = tf.gatherND(offsets, inds)

    //score: N x 1
    score = tf.expandDims(tf.gatherND(probs, inds), 1)

    // P-Net is applied to scaled images
    // so we need to rescale bounding boxes back
    inds = tf.cast(inds, 'float32')

    // bounding boxes: N x 9
    a = tf.split(inds, inds.shape[1], 1)

    bounding_boxes = tf.concat([
        tf.expandDims(tf.round(tf.mul(a[1].reshape([-1]), stride).div(scale)), 1),
        tf.expandDims(tf.round(tf.mul(a[0].reshape([-1]), stride).div(scale)), 1),
        tf.expandDims(tf.round(tf.mul(a[1].reshape([-1]), stride).add(cell_size).div(scale)), 1),
        tf.expandDims(tf.round(tf.mul(a[0].reshape([-1]), stride).add(cell_size).div(scale)), 1),
        score, offsets
    ], 1)
    
    return bounding_boxes

}

module.exports = {convert_to_square, calibrate_box, preprocess, get_image_boxes, generate_boxes}
const tf = require('@tensorflow/tfjs');
const tfnode = require('@tensorflow/tfjs-node');
const jpeg_js = require('jpeg-js');
const fs = require('fs');

const {PNet, RNet, ONet} = require('./models');
const {calibrate_box, convert_to_square, get_image_boxes, generate_boxes, preprocess} = require('./box_utils');


DEF_THRESHOLDS = [0.7, 0.8, 0.9]
DEF_NMS_THRESHOLDS = [0.6, 0.6, 0.6]

class MTCNN{
    // Top level class for mtcnn detection. """"
    constructor(pnet_path, rnet_path, onter_path, min_face_size=20.0, thresholds=null, mns_thresholds=null, max_output_size=300){
        this.pnet = PNet(pnet_path)
        this.rnet = RNet(rnet_path)
        this.onet = ONet(onter_path)
        this.min_face_size = min_face_size
        this.thresholds = thresholds || DEF_THRESHOLDS
        this.mns_thresholds = mns_thresholds || DEF_NMS_THRESHOLDS
        this.max_output_size = max_output_size
        this.scale_cache = {}
    }

    async detect(img){
        // """Detect faces and facial landmarks on an image

        // Parameters:
        //     img: rgb image, numpy array of shape [h, w, 3]

        // Returns:
        //     bboxes: float tensor of shape [n, 4], face bounding boxes
        //     landmarks: float tensor of shape [n, 10], 5 facial landmarks,
        //                 first 5 numbers of array are x coords, last are y coords
        //     scores: float tensor of shape [n], confidence scores
        // """

        const height = img.shape[0]
        const width = img.shape[1]
        const scales = this.get_scale(height, width)
        // console.log(scales)
        var boxes = await this.stage_one(img, scales)
        console.log("uaua", boxes.shape)
        if (boxes.shape[0] == 0){
            return []
        }
        boxes = await this.stage_two(img, boxes, height, width, boxes.shape[0])
        if (boxes.shape[0] == 0){
            return []
        }
        const data = await this.stage_three(img, boxes, height, width, boxes.shape[0])
        console.log('oiuio', data)
        const boxess = data['boxes']
        const landmarks = data['landmarks']
        const scores = data['scores']
        return {'boxes': boxess, 'landmarks': landmarks, 'scores': scores}
    }

    get_scale(height, width){
        // """Compute scaling factors for given image dimensions

        // Parameters:
        //     height: float
        //     width: float

        // Returns:
        //     list of floats, scaling factors
        // """
        // console.log(height)
        var min_length = Math.min(height, width)

        if (min_length in this.scale_cache){
            return this.scale_cache[min_length]
        }
        const min_detection_size = 12.0
        const factor = 0.707
        const scales = []

        const m = min_detection_size / this.min_face_size
        // min_length = tf.mul(min_length, m)
        min_length *= m
        var factor_count = 0
        // if (tf.greater(0, min_length)){
        //     console.log('this is min length', min_length)
        // }
        // min_length.print()
        // console.log(min_length.dataSync() < min_detection_size)
        while (min_length > min_detection_size){
            scales.push(m * factor ** factor_count)
            min_length = min_length * factor
            factor_count += 1
        }
        this.scale_cache[min_length] = scales
        // console.log(scales)
        return scales
    }

    // input_signature=[tf.tensor(shape=[null, null, 3], dtype='float32'),
    //                  tf.tensor(shape=[], dtype='float32'),
    //                  tf.tensor(shape=[], dtype='float32'),
    //                  tf.tensor(shape=[], dtype='float32')]
    
    async stage_one_scale(img, height, width, scale) {
        // """Perform stage one part with a given scaling factor

        // Parameters:
        //     img: rgb image, float tensor of shape [h, w, 3]
        //     height: image height, float
        //     width: image width, float
        //     scale: scaling factor, float

        // Returns:
        //     float tensor of shape [n, 9]
        // """
        var hs = tf.ceil(tf.mul(height, scale))
        var ws = tf.ceil(tf.mul(width, scale))

        var img_in = tf.image.resizeBilinear(img, [hs.dataSync()[0], ws.dataSync()[0]])
        img_in = preprocess(img_in)
        img_in = tf.expandDims(img_in, 0)
        const data = (await this.pnet).predict(img_in) // probs, offsets
        const probs = data[0]
        const offsets = data[1]
        // console.log('this oke',probs, offsets)
        // const cc = tf.slice(probs, [0, 0, 1], [-1, -1, 1])
        // probs[0] (1,49,79,2)
        // new_probs = tf.slice(probs, [0, 0, 1], [-1, -1, 1])

        // console.log('printttt', probs[0])
        const probs_zero = tf.tensor(probs.arraySync()[0])
        const offsets_zero = tf.tensor(offsets.arraySync()[0])
        const boxes = await generate_boxes(probs_zero, offsets_zero, scale, this.thresholds[0])
        if(boxes.shape[0] == 0){
            return boxes
        }
        // boxes[:, 0:4]
        // boxes[:, 4]
        
        // a = tf.split(boxes, 4, 1)
        const keep = tf.image.nonMaxSuppression(tf.slice(boxes, [0,0], [-1,4]), 
        tf.reshape(tf.slice(boxes, [0,3], [-1,1]), [-1]), 
        this.max_output_size, 0.5)
        return tf.gather(boxes, keep)
    }

    // input_signature = [tf.tensor(shape=[null, 9], dtype='float32')]
    stage_one_filter(boxes){
        // """Filter out boxes in stage one

        // Parameters:
        //     boxes: collected boxes with different scales, float tensor of shape [n, 9]

        // Returns:
        //     float tensor of shape [n, 4]
        // """

        // console.log("kooo", boxes.shape)
        var boxess = tf.slice(boxes, [0,0], [-1,4])
        const scores = tf.reshape(tf.slice(boxes, [0,3], [-1,1]), [-1])
        const offsets = tf.slice(boxes, [0,5], [-1,-1]) 


        boxess = calibrate_box(boxess, offsets)
        boxess = convert_to_square(boxess)
        const keep = tf.image.nonMaxSuppression(boxess, scores, this.max_output_size, this.mns_thresholds[0])
        
        boxess = tf.gather(boxess, keep)
        // console.log('aaaa', boxess.shape)
        return boxess
    }

    async stage_one(img, scales){
        // """Run stage one on the input image

        // Parameters:
        //     img: rgb image, float tensor of shape [h, w, 3]
        //     scales: scaling factors, list of floats

        // Returns:
        //     float tensor of shape [n, 4], predicted bounding boxes
        // """

        const height = img.shape[0]
        const width = img.shape[1]
        var boxes = []
        // console.log(scales)
        for (let i = 0; i < scales.length; i++){
            // console.log(i)
            boxes.push(await this.stage_one_scale(img, height, width, scales[i]))
        }

        boxes = await tf.concat(boxes, 0)
        if(boxes.shape[0] == 0){
            return []
        }
        
        return this.stage_one_filter(boxes)
    }
    // tf.TensorSpec(shape=(None, None, 3), dtype=tf.float32)
    // input_signature = [tf.TensorSpec(shape=[null, null, 3], dtype='float32'),
    //                     tf.tensor(shape=[null, 4], dtype='float32'),
    //                     tf.tensor(shape=[], dtype='float32'),
    //                     tf.tensor(shape=[], dtype='float32'),
    //                     tf.tensor(shape=[], dtype='int32')]

    async stage_two(img, boxes, height, width, num_boxes){
        // """Run stage two on the input image

        // Parameters:
        //     img: rgb image, float tensor of shape [h, w, 3]
        //     bboxes: bounding boxes from stage one, float tensor of shape [n, 4]
        //     height: image height, float
        //     width: image width, float
        //     num_boxes: number of rows in bboxes, int

        // Returns:
        //     float tensor of shape [n, 4], predicted bounding boxes
        // """
        
        var img_boxes = get_image_boxes(boxes, img, height, width, num_boxes, 24)
        const data =  (await this.rnet).predict(img_boxes)
        
        const probs = data[0]
        var offsets = data[1]

        //probs[:,1]
        const slice_probs = tf.reshape(tf.slice(probs, [0,1], [-1,1]), [-1])
        var keep = tf.reshape(tf.slice(await tf.whereAsync(tf.greater(slice_probs, this.thresholds[1])), [0,0], [-1,1]), [-1])

        boxes = tf.gather(boxes, keep)
        offsets = tf.gather(offsets, keep)
        const scores = tf.gather(tf.reshape(tf.slice(probs, [0,1], [-1,1]), [-1]), keep)
        console.log('that oke?', boxes.shape)
        boxes = calibrate_box(boxes, offsets)

        boxes = convert_to_square(boxes)
        keep = tf.image.nonMaxSuppression(boxes, scores, this.max_output_size, this.mns_thresholds[1])
        boxes = tf.gather(boxes, keep)
        return boxes
    }   

    // input_signature = [tf.tensor(shape=[null, null, 3], dtype='float32'),
    //                     tf.tensor(shape=[null, 4], dtype='float32'),
    //                     tf.tensor(shape=[], dtype='float32'),
    //                     tf.tensor(shape=[], dtype='float32'),
    //                     tf.tensor(shape=[], dtype='int32')]
    
    async stage_three(img, boxes, height, width, num_boxes){
        // """Run stage three on the input image

        // Parameters:
        //     img: rgb image, float tensor of shape [h, w, 3]
        //     bboxes: bounding boxes from stage two, float tensor of shape [n, 4]
        //     height: image height, float
        //     width: image width, float
        //     num_boxes: number of rows in bboxes, int

        // Returns:
        //     bboxes: float tensor of shape [n, 4], face bounding boxes
        //     landmarks: float tensor of shape [n, 10], 5 facial landmarks,
        //                 first 5 numbers of array are x coords, last are y coords
        //     scores: float tensor of shape [n], confidence scores
        // """

        const img_boxes = get_image_boxes(boxes, img, height, width, num_boxes, 48)
        
        const data = (await this.onet).predict(img_boxes)

        const probs = data[0]
        var offsets = data[1]
        var landmarks = data[2] 

        //probs[:,1]
        const slice_probs = tf.reshape(tf.slice(probs, [0,1], [-1,1]), [-1])
        var keep = tf.reshape(tf.slice(await tf.whereAsync(tf.greater(slice_probs, this.thresholds[2])), [0,0], [-1,1]), [-1])

        boxes = tf.gather(boxes, keep)
        offsets = tf.gather(offsets, keep)
        var scores = tf.gather(tf.reshape(tf.slice(probs, [0,1], [-1,1]), [-1]), keep)
        landmarks = tf.gather(landmarks, keep)

        //compute landmak points

        width = tf.expandDims(tf.reshape(tf.slice(boxes, [0,2], [-1,1]), [-1]).sub(tf.reshape(tf.slice(boxes, [0,0], [-1,1]), [-1])).add(1.0), 1)
        height = tf.expandDims(tf.reshape(tf.slice(boxes, [0,3], [-1,1]), [-1]).sub(tf.reshape(tf.slice(boxes, [0,1], [-1,1]), [-1])).add(1.0), 1)
        const xmin = tf.expandDims(tf.reshape(tf.slice(boxes, [0,0], [-1,1]), [-1]), 1)
        const ymin = tf.expandDims(tf.reshape(tf.slice(boxes, [0,1], [-1,1]), [-1]), 1)
        landmarks = tf.concat([
                        tf.mul(tf.slice(landmarks, [0,0], [-1,5]), width).add(xmin), 
                        tf.mul(tf.slice(landmarks, [0,5], [-1,5]), height).add(ymin)
                    ], 1)
        boxes = calibrate_box(boxes, offsets)
        keep = tf.image.nonMaxSuppression(boxes, scores, this.max_output_size, this.mns_thresholds[2])
        boxes = tf.gather(boxes, keep)
        scores = tf.gather(scores, keep)
        landmarks = tf.gather(landmarks, keep)
        return {'boxes': boxes, 'landmarks': landmarks, 'scores': landmarks}
    }   
}

module.exports = MTCNN
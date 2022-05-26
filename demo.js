const tf = require('@tensorflow/tfjs');
const tfnode = require('@tensorflow/tfjs-node');
require('@tensorflow/tfjs-node');
// const jpeg_js = require('jpeg-js');
const fs = require('fs');

var MTCNN = require('./mtcnn');
// import MTCNN from './mtcnn';

async function image_demo(img_url, img_output){
    // """mtcnn image demo"""
    mtcnn = new MTCNN('file:///home/whoisltd/works/mtcnn-tfjs/my_models/pnet/model.json', 
                    'file:///home/whoisltd/works/mtcnn-tfjs/my_model/rnet/model.json', 
                    'file:///home/whoisltd/works/mtcnn-tfjs/my_model/onet/model.json');
    
    img = fs.readFileSync("/home/whoisltd/Desktop/as4.jpg");
    var tensor = tfnode.node.decodeImage(img)
 
    // fs.writeFileSync(img_output, tensor);
    
    // img = jpeg_js.decode(fs.readFileSync(img_url));
    // img = tf.browser.fromPixels(img);
    //convert color bgr to rgb
    // tensor.print()
    // tensor1 = tf.reverse(tensor, 1);
    data = await mtcnn.detect(tensor);
    // console.log('uiii', data)
    const boxes = data['boxes'].arraySync()[0]
    for (let i = 0; i < 4; i++){
        boxes[i] = Math.round(boxes[i])
    }
    //crop image 
    // cropped = img[y:y+h, x:x+w]
    // x, y, w, h = face['box']
    // const row = boxes[]
    console.log(boxes)
    console.log(tensor.shape)
    var tensor1 = tf.slice(tensor, [boxes[1], boxes[0]], [boxes[3]-boxes[1], boxes[2]-boxes[0]])
    // var tensor1 = draw_faces(tensor, data['boxes'], data['landmarks'], data['scores']);
    tensor1 = tfnode.node.encodeJpeg(tensor1).then((f) => { 2
        fs.writeFileSync("simple.jpg", f)});
    
}

function draw_faces(img, boxes, landmarks, scores){
    //"""Draw bounding boxes and landmarks on image"""
    // console.log('ddd',boxes)
    
    for (let i = 0; i < boxes.lenght; i++){
        const box = boxes.slice([i, 0], [1, 4]);
        // const landmark = landmarks.slice([i, 0], [1, 10]);
        // const score = scores.slice([i, 0], [1, 1]);
        // const color = tf.tensor1d([0, 0, 255]).mul(tf.expandDims(score, -1));
        img = tf.image.cropAndResize(img, box, tf.range(0, 1), [48, 48]).print();
        // img = tf.image.drawBoundingBoxes(img, tf.expandDims(box, 0));
        // img = tf.image.drawKeypoints(img, tf.expandDims(landmark, 0), {color: color});
    }
    return img;
}

image_demo('/home/whoisltd/Desktop/as6.jpg', 'output.png');
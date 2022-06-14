const tf = require('@tensorflow/tfjs-node');
const sharp = require('sharp');
const fs = require('fs');
var MTCNN = require('./mtcnn');
const canvas = require('canvas');
const path = require('path');

async function detect(img_url, draw = false, crop = false) {
    // """mtcnn image demo"""
    mtcnn = new MTCNN('file://' + path.join(__dirname, 'final_model/pnet/model.json'),
    'file://' + path.join(__dirname, 'final_model/rnet/model.json'), 
    'file://' + path.join(__dirname, 'final_model/onet/model.json'));
    
    try {
        var img = await sharp(img_url).rotate().toBuffer()
    } catch (error) {
        console.log(error);
        return;
    }
    
    try {
        var tensor = tf.node.decodeImage(img)
    } catch (error) {
        console.log("Error: " + error);
        return;
    }
    
    data = await mtcnn.detect(tensor);

    const boxes = data['boxes'].arraySync()[0]
    const landmarks = data['landmarks'].arraySync()[0]
    const scores = data['scores'].arraySync()[0]

    // """Draw bounding boxes on image"""
    if (draw){
        canvas.loadImage(img).then(image => {
            const ct = canvas.createCanvas(image.width, image.height)
            const ctx = ct.getContext('2d');
            
            ctx.drawImage(image, 0, 0);
            ctx.beginPath()
            ctx.rect(boxes[0], boxes[1], boxes[2]-boxes[0], boxes[3]-boxes[1])
            ctx.strokeStyle = 'red'
            ctx.lineWidth = 4
            ctx.stroke()
            ctx.closePath()

            var img_out = ct.toBuffer('image/jpeg')
    
            fs.writeFileSync('draw.jpg', img_out)
        })
    }

    // crop face
    if (crop){
        for (let i = 0; i < 4; i++){
            boxes[i] = Math.round(boxes[i])
        }
    
        var cropped = tf.slice(tensor, [boxes[1], boxes[0]], [boxes[3]-boxes[1], boxes[2]-boxes[0]])
    
        tf.node.encodeJpeg(cropped).then((f) => {
            fs.writeFileSync("cropped.jpg", f)});
    }
    const dict = {}
    dict['boxes'] = boxes
    dict['landmarks'] = landmarks
    dict['scores'] = scores
    return dict
}

exports.detect = detect;
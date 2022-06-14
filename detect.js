const tf = require('@tensorflow/tfjs-node');
const sharp = require('sharp');
const fs = require('fs');
var MTCNN = require('./mtcnn');
const canvas = require('canvas');
const path = require('path');

async function detect(img_url, img_output, crop = false) {
    // """mtcnn image demo"""
    mtcnn = new MTCNN('file://' + path.join(__dirname, 'final_model/pnet/model.json'),
    'file://' + path.join(__dirname, 'final_model/rnet/model.json'), 
    'file://' + path.join(__dirname, 'final_model/onet/model.json'));
    
    try {
        const img = await sharp(img_url).rotate().toBuffer()
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

        if (img_output == null){
            img_output = 'output.jpg'
        }
        var fileExt = img_output.split('.').pop();
        var img_out = null
        if (fileExt == 'jpg' || fileExt == 'jpeg') {
            img_out = ct.toBuffer('image/jpeg')
        } else if (fileExt == 'png') {
            img_out = ct.toBuffer('image/png')
        } else{
            return "Error: unsupported image output format"
        }

        fs.writeFileSync(img_output, img_out)
    })

    // crop face
    if (crop == true){
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

export default detect
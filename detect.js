const tf = require('@tensorflow/tfjs-node');
const sharp = require('sharp');
const fs = require('fs');
var MTCNN = require('./mtcnn');
const canvas = require('canvas');
const url = 'https://digitalwallet-poc-storage-s3.s3.ap-southeast-1.amazonaws.com/ai_temp/mtcnn/'
const pnet_url = url + 'pnet/model.json'
const rnet_url = url + 'rnet/model.json'
const onet_url = url + 'onet/model.json'

function load_model()
{
    mtcnn = new MTCNN(pnet_url, rnet_url, onet_url);
    return mtcnn
}

var mtcnn = load_model()

async function load_img(img_url){
    try {
        var img = await sharp(img_url).rotate().toBuffer()
    } catch (error) {
        console.log(error);
        return;
    }
    
    try {
        var tensor = tf.node.decodeImage(img)
    } catch (error) {
        console.log(error);
        return;
    }
    return {img, tensor}
}
async function draw_img(img_url, output_url=null) {

    const data = await detect(img_url)
    var {img} = await load_img(img_url)
    const boxes = data.boxes

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

        
        if (output_url == null){
            var img_out = ct.toBuffer('image/jpeg')
            fs.writeFileSync('draw.jpg', img_out)
            return;
        }

        var fileExt = output_url.split('.').pop();
        var img_out = null
        if (fileExt == 'jpg' || fileExt == 'jpeg') {
            img_out = ct.toBuffer('image/jpeg')
        }
        else if (fileExt == 'png'){
            img_out = ct.toBuffer('image/png')
        }
        else {
            console.log('Output type is not supported')
            return;
        }
        fs.writeFileSync(output_url, img_out)
        
    })

}
async function crop_face(img_url, output_url = null, return_img = false) {
    const data = await detect(img_url)
    var {tensor} = await load_img(img_url)
    const boxes = data.boxes

    // crop face

    for (let i = 0; i < 4; i++){
        boxes[i] = Math.round(boxes[i])
    }

    var cropped = tf.slice(tensor, [boxes[1], boxes[0]], [boxes[3]-boxes[1], boxes[2]-boxes[0]])
    if (return_img){
        if (output_url != null){
            tf.node.encodeJpeg(cropped).then((f) => {
                fs.writeFileSync(output_url, f)});
        }
        else{
            tf.node.encodeJpeg(cropped).then((f) => {
                fs.writeFileSync('cropped.jpg', f)});
        }
    }
    return cropped

}
async function detect(img_url) {
    // """mtcnn image demo"""
    
    var {tensor} = await load_img(img_url)
    
    data = await mtcnn.detect(tensor);

    const {boxes, landmarks, scores} = data

    const dict = {}
    dict['boxes'] = boxes
    dict['landmarks'] = landmarks
    dict['scores'] = scores
    return dict
}

// exports.detect = detect;
// exports.draw_img = draw_img;
// exports.crop_face = crop_face;
draw_img('/home/whoisltd/Desktop/as12.jpg')
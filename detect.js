const tf = require('@tensorflow/tfjs-node');
const sharp = require('sharp');
const fs = require('fs');
var mtcnn_js = require('./mtcnn');
const canvas = require('canvas');
 
// const pnet_url = 'https://storage.googleapis.com/my-mtcnn-models/final_model/pnet/model.json'
// const rnet_url = 'https://storage.googleapis.com/my-mtcnn-models/final_model/rnet/model.json'
// const onet_url = 'https://storage.googleapis.com/my-mtcnn-models/final_model/onet/model.json'

class MTCNN {
    constructor(pnet, rnet, onet) {
        this.mtcnn = this.load_model(pnet, rnet, onet)
    }

    load_model(pnet, rnet, onet)
    {
        var model = new mtcnn_js(pnet, rnet, onet);
        return model
    }

    async load_img(img_url){
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
    async draw_img(img_url, output_url=null) {

        const data = await this.detect(img_url)
        var {img} = await this.load_img(img_url)
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
    async crop_face(img_url, output_url = null, return_img = false) {
        const data = await this.detect(img_url)
        var {tensor} = await this.load_img(img_url)
        const boxes = data.boxes

        // crop face

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
    async detect(img_url) {
        // """mtcnn image demo"""
        
        var {tensor} = await this.load_img(img_url)
        
        var data = await this.mtcnn.detect(tensor);

        const {boxes, landmarks, scores} = data

        for (let i = 0; i < 4; i++){
            if (boxes[i] < 0){
                boxes[i] = 0
            }
        }

        const dict = {}
        dict['boxes'] = boxes
        dict['landmarks'] = landmarks
        dict['scores'] = scores
        return dict
    }
}

exports.MTCNN = MTCNN;

// const pnet_url = 'https://storage.googleapis.com/my-mtcnn-models/final_model/pnet/model.json'
// const rnet_url = 'https://storage.googleapis.com/my-mtcnn-models/final_model/rnet/model.json'
// const onet_url = 'https://storage.googleapis.com/my-mtcnn-models/final_model/onet/model.json'
// mtn = new MTCNN(pnet_url, rnet_url, onet_url)

// mtn.crop_face('/home/whoisltd/Desktop/dat.jpg', null, true)
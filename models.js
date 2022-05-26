const { models } = require('@tensorflow/tfjs');
const tf = require('@tensorflow/tfjs')
require('@tensorflow/tfjs-node')

async function PNet(model_url = null){
    // """ Proposal Network, receives an image and outputs
    // bbox offset regressions and confidence scores for each sliding
    // window of 12x12
    // """

    // img = tf.input({shape: [null, null, 3]})
    // // x = layers.Permute((2, 1, 3), name='permute')(img_in) to tfjs
    // x = tf.layers.permute({dims: [2, 1, 3]}).apply(img)
    // // x = layers.Conv2D(10, 3, 1, name='conv1')(x)
    // x = tf.layers.conv2d({filters: 10, kernelSize: 3, strides: 1, padding: 'same', name: 'conv1'}).apply(x)
    // // x = layers.PReLU(shared_axes=[1, 2], name='prelu1')(x)
    // x = tf.layers.prelu({sharedAxes: [1, 2], name: 'prelu1'}).apply(x)
    // // x = layers.MaxPool2D(2, 2, padding='same', name='pool1')(x)
    // x = tf.layers.maxPooling2d({poolSize: 2, strides: 2, padding: 'same', name: 'pool1'}).apply(x)

    // // x = layers.Conv2D(16, 3, 1, name='conv2')(x)
    // x = tf.layers.conv2d({filters: 16, kernelSize: 3, strides: 1, padding: 'same', name: 'conv2'}).apply(x)
    // // x = layers.PReLU(shared_axes=[1, 2], name='prelu2')(x)
    // x = tf.layers.prelu({sharedAxes: [1, 2], name: 'prelu2'}).apply(x)

    // // x = layers.Conv2D(32, 3, 1, name='conv3')(x)
    // x = tf.layers.conv2d({filters: 32, kernelSize: 3, strides: 1, padding: 'same', name: 'conv3'}).apply(x)
    // // x = layers.PReLU(shared_axes=[1, 2], name='prelu3')(x)
    // x = tf.layers.prelu({sharedAxes: [1, 2], name: 'prelu3'}).apply(x)

    // a = tf.layers.conv2d({filters: 2, kernelSize: 1, strides: 1, padding: 'same', name: 'conv4_1'}).apply(x)
    // a = tf.layers.softmax().apply(a)
    
    // // a = tf.transpose(a, [0, 2, 1, 3])
    // a = tf.layers.reshape({targetShape: [null, null, 2]}).apply(a)
    // b = tf.layers.conv2d({filters: 4, kernelSize: 1, strides: 1, padding: 'same', name: 'conv4_2'}).apply(x)

    // // b = tf.transpose(b, [0, 2, 1, 3])
    // b = tf.layers.reshape({targetShape: [null, null, 4]}).apply(b)
    

    // model = tf.model({inputs: img, outputs: [a, b]})
    if (model_url != null){
        // model.load_weights(model_url)
        const model = await tf.loadLayersModel(model_url);

        return model
    }
    // return model
} 

async function RNet(model_url = null){
    // """ Refine Network, receives image crops from PNet and outputs
    // further offset refinements and confidence scores to filter out
    // the predictions
    // """

    // img = tf.input({shape: [24, 24, 3]})

    // // x = layers.Permute((2, 1, 3), name='permute')(img_in)
    // x = tf.layers.permute({dims: [2, 1, 3]}).apply(img)
    // // x = layers.Conv2D(28, 3, 1, name='conv1')(x)
    // x = tf.layers.conv2d({filters: 28, kernelSize: 3, strides: 1, padding: 'same', name: 'conv1'}).apply(x)
    // // x = layers.PReLU(shared_axes=[1, 2], name='prelu1')(x)
    // x = tf.layers.prelu({sharedAxes: [1, 2], name: 'prelu1'}).apply(x)
    // // x = layers.MaxPool2D(3, 2, padding='same', name='pool1')(x)
    // x = tf.layers.maxPooling2d({poolSize: 3, strides: 2, padding: 'same', name: 'pool1'}).apply(x)

    // // x = layers.Conv2D(48, 3, 1, name='conv2')(x)
    // x = tf.layers.conv2d({filters: 48, kernelSize: 3, strides: 1, padding: 'same', name: 'conv2'}).apply(x)
    // // x = layers.PReLU(shared_axes=[1, 2], name='prelu2')(x)
    // x = tf.layers.prelu({sharedAxes: [1, 2], name: 'prelu2'}).apply(x)
    // // x = layers.MaxPool2D(3, 2, padding='same', name='pool2')(x)
    // x = tf.layers.maxPooling2d({poolSize: 3, strides: 2, padding: 'same', name: 'pool2'}).apply(x)

    // // x = layers.Conv2D(64, 2, 1, name='conv3')(x)
    // x = tf.layers.conv2d({filters: 64, kernelSize: 2, strides: 1, padding: 'same', name: 'conv3'}).apply(x)
    // // x = layers.PReLU(shared_axes=[1, 2], name='prelu3')(x)
    // x = tf.layers.prelu({sharedAxes: [1, 2], name: 'prelu3'}).apply(x)

    // // x = layers.Flatten(name='flatten')(x)
    // x = tf.layers.flatten({name: 'flatten'}).apply(x)
    // // x = layers.Dense(128, name='conv4')(x)
    // x = tf.layers.dense({units: 128, name: 'conv4'}).apply(x)
    // // x = layers.PReLU(name='prelu4')(x)
    // x = tf.layers.prelu({name: 'prelu4'}).apply(x)

    // //a = layers.Dense(2, name='conv5_1')(x)
    // a = tf.layers.dense({units: 2, name: 'conv5_1'}).apply(x)
    // // a = layers.Softmax(name='prob1')(a)
    // a = tf.layers.softmax({name: 'prob1'}).apply(a)
    // //b = layers.Dense(4, name='conv5_2')(x)
    // b = tf.layers.dense({units: 4, name: 'conv5_2'}).apply(x)
    // const model = await tf.loadLayersModel(model_url);
    // return model
    // const model = tf.model({inputs: img, outputs: [a, b]})
    // model.loadWeights('/home/whoisltd/works/mtcnn-tfjs/weights/rnet/model.json')
    // model.summary()
    if (model_url != null){
        // model.load_weights(model_url)
        const model = await tf.loadLayersModel(model_url);

        return model
    }

    
}

async function ONet(model_url = null){
    // """ Output Network, receives image crops from RNet and outputs
    // final offset regressions, facial landmark positions and confidence scores
    // """
    // img = tf.input({shape: [48, 48, 3]})

    // // x = layers.Permute((2, 1, 3), name='permute')(img_in)
    // x = tf.layers.permute({dims: [2, 1, 3]}).apply(img)
    // // x = layers.Conv2D(32, 3, 1, name='conv1')(x)
    // x = tf.layers.conv2d({filters: 32, kernelSize: 3, strides: 1, padding: 'same', name: 'conv1'}).apply(x)
    // // x = layers.PReLU(shared_axes=[1, 2], name='prelu1')(x)
    // x = tf.layers.prelu({sharedAxes: [1, 2], name: 'prelu1'}).apply(x)
    // // x = layers.MaxPool2D(3, 2, padding='same', name='pool1')(x)
    // x = tf.layers.maxPooling2d({poolSize: 3, strides: 2, padding: 'same', name: 'pool1'}).apply(x)

    // // x = layers.Conv2D(64, 3, 1, name='conv2')(x)
    // x = tf.layers.conv2d({filters: 64, kernelSize: 3, strides: 1, padding: 'same', name: 'conv2'}).apply(x)
    // // x = layers.PReLU(shared_axes=[1, 2], name='prelu2')(x)
    // x = tf.layers.prelu({sharedAxes: [1, 2], name: 'prelu2'}).apply(x)
    // // x = layers.MaxPool2D(3, 2, padding='same', name='pool2')(x)
    // x = tf.layers.maxPooling2d({poolSize: 3, strides: 2, padding: 'same', name: 'pool2'}).apply(x)

    // // x = layers.Conv2D(64, 3, 1, name='conv3')(x)
    // x = tf.layers.conv2d({filters: 64, kernelSize: 3, strides: 1, padding: 'same', name: 'conv3'}).apply(x)
    // // x = layers.PReLU(shared_axes=[1, 2], name='prelu3')(x)
    // x = tf.layers.prelu({sharedAxes: [1, 2], name: 'prelu3'}).apply(x)
    // // x = layers.MaxPool2D(2, 2, padding='same', name='pool3')(x)
    // x = tf.layers.maxPooling2d({poolSize: 2, strides: 2, padding: 'same', name: 'pool3'}).apply(x)

    // // x = layers.Conv2D(128, 2, 1, name='conv4')(x)
    // x = tf.layers.conv2d({filters: 128, kernelSize: 2, strides: 1, padding: 'same', name: 'conv4'}).apply(x)
    // // x = layers.PReLU(shared_axes=[1, 2], name='prelu4')(x)
    // x = tf.layers.prelu({sharedAxes: [1, 2], name: 'prelu4'}).apply(x)

    // // x = layers.Flatten(name='flatten')(x)
    // x = tf.layers.flatten({name: 'flatten'}).apply(x)
    // // x = layers.Dense(256, name='conv5')(x)
    // x = tf.layers.dense({units: 256, name: 'conv5'}).apply(x)
    // // x = layers.PReLU(name='prelu5')(x)
    // x = tf.layers.prelu({name: 'prelu5'}).apply(x)

    // // a = layers.Dense(2, name='conv6_1')(x)
    // a = tf.layers.dense({units: 2, name: 'conv6_1'}).apply(x)
    // // a = layers.Softmax(name='prob1')(a)
    // a = tf.layers.softmax({name: 'prob1'}).apply(a)
    // // b = layers.Dense(4, name='conv6_2')(x)
    // b = tf.layers.dense({units: 4, name: 'conv6_2'}).apply(x)
    // // c = layers.Dense(10, name='conv6_3')(x)
    // c = tf.layers.dense({units: 10, name: 'conv6_3'}).apply(x)

    // model = tf.model({inputs: img, outputs: [a, b, c]})
    if (model_url != null){
        // model.load_weights(model_url)
        const model = await tf.loadLayersModel(model_url);
        // const model = await tf.loadModel(model_url)
        return model
    }
    
}

// async function main(){
//     const model = await tf.loadGraphModel(
// 'file:///home/whoisltd/works/mtcnn-tfjs/weights/onet/model.json');
//     model.summary();
//     // a = model.predict(tf.zeros([1, 48, 48, 3]));
//     // console.log(a);
//     return model
// }
// main()
a = RNet()
// b = a.predict(tf.zeros([1, 24, 24, 3]))
// b[0].print()
// b[1].print()
// async function main2(){
//     a = await main()
//     b = await a.predict(tf.zeros([1, 48, 48, 3]));
//     // console.log('etst');
//     console.log(b);
// }
// main2()
// class b{
//     constructor(){
//         this.a = main()
//     }
//     async test(){
//         const bc = (await this.a).predict(tf.zeros([1, 48, 48, 3]));
//         console.log(bc);
//     }
// }
// c = new b()
// c.test()

module.exports = {PNet, RNet, ONet}
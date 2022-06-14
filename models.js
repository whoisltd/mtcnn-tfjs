const tf = require('@tensorflow/tfjs-node')

async function PNet(model_url = null){
    // """ Proposal Network, receives an image and outputs
    // bbox offset regressions and confidence scores for each sliding
    // window of 12x12
    // """

    if (model_url != null){
        const model = await tf.loadLayersModel(model_url);
        return model
    }
    return "missing PNet model url"
} 

async function RNet(model_url = null){
    // """ Refine Network, receives image crops from PNet and outputs
    // further offset refinements and confidence scores to filter out
    // the predictions
    // """

    if (model_url != null){
        const model = await tf.loadLayersModel(model_url);
        return model
    }
    return "missing RNet model url"

}

async function ONet(model_url = null){
    // """ Output Network, receives image crops from RNet and outputs
    // final offset regressions, facial landmark positions and confidence scores
    // """

    if (model_url != null){
        const model = await tf.loadLayersModel(model_url);
        return model
    }

    return "missing ONet model url"
    
}

module.exports = {PNet, RNet, ONet}
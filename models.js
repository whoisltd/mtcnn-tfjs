const tf = require('@tensorflow/tfjs-node')

async function Model(model_url = null){
    // """ Proposal Network, receives an image and outputs
    // bbox offset regressions and confidence scores for each sliding
    // window of 12x12
    // """

    // """ Refine Network, receives image crops from PNet and outputs
    // further offset refinements and confidence scores to filter out
    // the predictions
    // """

    // """ Output Network, receives image crops from RNet and outputs
    // final offset regressions, facial landmark positions and confidence scores
    // """

    if (model_url != null){
        try {
            const model = await tf.loadLayersModel(model_url);
            return model
        } catch (error) {
            console.error('You need to connect to internet')

        }

    }
    return "missing model url"
} 

module.exports = Model
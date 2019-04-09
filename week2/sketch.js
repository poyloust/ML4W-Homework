let video;
const featureExtractor = ml5.featureExtractor('MobileNet',gotModel);;
const knnClassifier = ml5.KNNClassifier();


function setup(){
    video = createCapture(VIDEO);
    video.parent('videoContainer');

    createButtons();
}


function gotModel(){
    console.log('model Loaded');
}


function createButtons(){
    buttonA = select('#addClassRock');
    buttonA.mousePressed(function(){
        addExample('Rock');
    })
    buttonB = select('#addClassPaper');
    buttonB.mousePressed(function(){
        addExample('Paper');
    })
    buttonC = select('#addClassScissor');
    buttonC.mousePressed(function(){
        addExample('Scissor');
    })
    buttonClassify = select('#buttonPredict');
    buttonClassify.mousePressed(function(){
        classifyVideo();
    })
}

function classifyVideo(){
    const features = featureExtractor.infer(video);  //infer is to get high level features
    knnClassifier.classify(features, gotResult);
}

function gotResult(err, results){
    console.log('results:',results)
    classifyVideo();
}

function addExample(label){
    const features = featureExtractor.infer(video);  //infer is to get high level features
    knnClassifier.addExample(features,label);
}

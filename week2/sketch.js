let video;
const knnClassifier = ml5.KNNClassifier();
let poseNet;
let poses = [];
var rabbit = false;
var pig = false;

function preload(){
    r_l_eye = loadImage('img/l_eye.png');
    r_r_eye = loadImage('img/r_eye.png');
    r_nose = loadImage('img/mouth.png');
    p_l_ear = loadImage('img/l_ear.png');
    p_r_ear = loadImage('img/r_ear.png');
    p_nose = loadImage('img/nose.png');
}
function setup(){

    var canvas = createCanvas(640, 480);
    canvas.parent('videoContainer');
    video = createCapture(VIDEO);
    video.size(width, height);
    createButtons();

    poseNet = ml5.poseNet(video, modelReady);
    poseNet.on('pose', function(results) {
        poses = results;
  });
  video.hide();
}

function draw(){
    image(video, 0, 0, width, height);
    drawFace();
    //drawKeypoints();
    drawSkeleton();
}



function modelReady(){
    console.log('posenet model loaded');
}


function createButtons(){
    buttonA = select('#addClassA');
    buttonA.mousePressed(function(){
        addExample('A');
    });
    buttonB = select('#addClassB');
    buttonB.mousePressed(function(){
        addExample('B');
    });
    resetBtnA = select('#resetA');
    resetBtnA.mousePressed(function() {
      clearClass('A');
    });
      
    resetBtnB = select('#resetB');
    resetBtnB.mousePressed(function() {
      clearClass('B');
    });

    buttonPredict = select('#buttonPredict');
    buttonPredict.mousePressed(classify);

    // Clear all classes button
    buttonClearAll = select('#clearAll');
    buttonClearAll.mousePressed(clearAllClasses);


}

function gotResults(err, result){
    if (err) {
        console.error(err);
      }
    
      if (result.confidencesByLabel) {
        const confideces = result.confidencesByLabel;
        // result.label is the label that has the highest confidence
        if (result.label) {
          select('#result').html(result.label);
          select('#confidence').html(`${confideces[result.label] * 100} %`);
        }
        if(confideces['A'] > 0.7){
            rabbit = true;
        }
        else{
            rabbit = false;
        }        
        if(confideces['B'] > 0.7){
            pig = true;
        }
        else{
            pig = false;
        }
        select('#confidenceA').html(`${confideces['A'] ? confideces['A'] * 100 : 0} %`);
        select('#confidenceB').html(`${confideces['B'] ? confideces['B'] * 100 : 0} %`);
      }
    
      classify();
}
function addExample(label) {
    // Convert poses results to a 2d array [[score0, x0, y0],...,[score16, x16, y16]]
    const poseArray = poses[0].pose.keypoints.map(p => [p.score, p.position.x, p.position.y]);
  
    // Create a tensor2d from 2d array
    const logits = ml5.tf.tensor2d(poseArray);
  
    // Add an example with a label to the classifier
    knnClassifier.addExample(logits, label);
    updateExampleCounts();

  }
  
  function classify() {
    const numClasses = knnClassifier.getNumClasses();
    if (numClasses <= 0) {
      console.error('There is no examples in any class');
      return;
    }
    // Convert poses results to a 2d array [[score0, x0, y0],...,[score16, x16, y16]]
    const poseArray = poses[0].pose.keypoints.map(p => [p.score, p.position.x, p.position.y]);
  
    // Create a tensor2d from 2d array
    const logits = ml5.tf.tensor2d(poseArray);
  
    // Use knnClassifier to classify which class do these features belong to
    // You can pass in a callback function `gotResults` to knnClassifier.classify function
    knnClassifier.classify(logits, gotResults);
  }
  function updateExampleCounts() {
    const counts = knnClassifier.getClassExampleCountByLabel();
  
    select('#exampleA').html(counts['A'] || 0);
    select('#exampleB').html(counts['B'] || 0);
  }
  
  // Clear the examples in one class
  function clearClass(classLabel) {
    knnClassifier.clearClass(classLabel);
    updateExampleCounts();
  }
  
  // Clear all the examples in all classes
  function clearAllClasses() {
    knnClassifier.clearAllClasses();
    updateExampleCounts();
  }

  function drawFace(){

    strokeWeight(2);

    if (poses.length > 0) {
        let pose = poses[0].pose.keypoints;
        // Create a pink ellipse for the nose
        fill(213, 0, 143);;
        let nose = pose[0].position;
        let r_eye = pose[1].position;
        let l_eye = pose[2].position;
        let r_ear = pose[3].position;
        let l_ear = pose[4].position;
        
        if(rabbit == true){
            image(r_r_eye,r_eye.x-20,r_eye.y-36,80,80);
            image(r_l_eye,l_eye.x-45,l_eye.y-40,80,80);
            image(r_nose, nose.x-40, nose.y-45, 100, 100);
        }
        if(pig == true){
            image(p_r_ear,r_ear.x-20,r_ear.y-80,50,50);
            image(p_l_ear,l_ear.x-35,l_ear.y-80,50,50);
            image(p_nose, nose.x-38, nose.y-35, 70, 60);
        }
    }
}

function drawKeypoints()  {
    // Loop through all the poses detected
    for (let i = 0; i < poses.length; i++) {
      // For each pose detected, loop through all the keypoints
      let pose = poses[i].pose;
      for (let j = 0; j < pose.keypoints.length; j++) {
        // A keypoint is an object describing a body part (like rightArm or leftShoulder)
        let keypoint = pose.keypoints[j];
        // Only draw an ellipse is the pose probability is bigger than 0.2
        if (keypoint.score > 0.2) {
          fill(255, 0, 0);
          noStroke();
          ellipse(keypoint.position.x, keypoint.position.y, 10, 10);
        }
      }
    }
  }
  
  // A function to draw the skeletons
  function drawSkeleton() {
    // Loop through all the skeletons detected
    for (let i = 0; i < poses.length; i++) {
      let skeleton = poses[i].skeleton;
      // For every skeleton, loop through all body connections
      for (let j = 0; j < skeleton.length; j++) {
        let partA = skeleton[j][0];
        let partB = skeleton[j][1];
        stroke(255, 0, 0);
        line(partA.position.x, partA.position.y, partB.position.x, partB.position.y);
      }
    }
  }
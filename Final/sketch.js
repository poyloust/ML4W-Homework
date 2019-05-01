let video;
const knnClassifier = ml5.KNNClassifier();
let poseNet;
let poses = [];

var rabbit = false;
var pig = false;
var currentResult = 0;
var pResult;
let style1;
let style2;
let resultImg;

function setup() {

  var canvas = createCanvas(640, 480);
  canvas.parent('videoContainer');
  video = createCapture(VIDEO);
  video.size(width, height);
  createButtons();
  poseNet = ml5.poseNet(video, modelReady);
  poseNet.on('pose', function (results) {
    poses = results;
  });
  video.hide();

  resultImg = createImg('');
  resultImg.hide();
  style1 = ml5.styleTransfer('models/picassoModel', video, modelALoaded);
  style2 = ml5.styleTransfer('models/vangoghModel', video, modelBLoaded);
}

function draw() {
  if (currentResult == 0) {
    console.log('show original vid');
    image(video, 0, 0, width, height);
    //drawFace();
    //drawKeypoints();
    drawSkeleton();
  }
}

//////////
//////////

function modelReady() {
  console.log('pose net model loaded');
}
function modelALoaded() {
  console.log('style transfer pop art model loaded');
}
function modelBLoaded() {
  console.log('style transfer renoir model loaded');
}

///////
///////

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
  // You can pass in a callback function `poseResults` to knnClassifier.classify function
  knnClassifier.classify(logits, poseResults);
}

/// this function need further optimise

function poseResults(err, result) {
  if (err) {
    console.error(err);
    classify();
  }

  if (result.confidencesByLabel) {
    const confideces = result.confidencesByLabel;
    // result.label is the label that has the highest confidence
    if (result.label) {
      select('#result').html(result.label);
      select('#confidence').html(`${confideces[result.label] * 100} %`);
    }
    if (confideces['A'] > 0.9) {
      rabbit = true;
      currentResult = 1;
    }
    else {
      rabbit = false;
    }
    if (confideces['B'] > 0.9) {
      pig = true;
      currentResult = 2;
    }
    else {
      pig = false;
    }
    if (pResult != currentResult) {
      console.log('change of state, now is ' + currentResult);
      if (currentResult == 1) {
        style1.transfer(style1Result);
      }
      if (currentResult == 2){
        style2.transfer(style2Result);
      }
    }
    //console.log (pResult, currentResult);
    select('#confidenceA').html(`${confideces['A'] ? confideces['A'] * 100 : 0} %`);
    select('#confidenceB').html(`${confideces['B'] ? confideces['B'] * 100 : 0} %`);
  }
  pResult = currentResult;
  classify();
}

/////////////
/////////////


function style1Result(err, img) {
  resultImg.attribute('src', img.src);
  image(resultImg, 0, 0, 640, 480);
  if(currentResult==1){
    style1.transfer(style1Result);
  }
}

function style2Result(err, img) {
  resultImg.attribute('src', img.src);
  image(resultImg, 0, 0, 640, 480);
  if(currentResult==2){
    style2.transfer(style2Result);
  }
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

function createButtons() {
  buttonA = select('#addClassA');
  buttonA.mousePressed(function () {
    addExample('A');
  });
  buttonB = select('#addClassB');
  buttonB.mousePressed(function () {
    addExample('B');
  });
  resetBtnA = select('#resetA');
  resetBtnA.mousePressed(function () {
    clearClass('A');
  });

  resetBtnB = select('#resetB');
  resetBtnB.mousePressed(function () {
    clearClass('B');
  });

  buttonPredict = select('#buttonPredict');
  buttonPredict.mousePressed(classify);

  // Clear all classes button
  buttonClearAll = select('#clearAll');
  buttonClearAll.mousePressed(clearAllClasses);
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


function drawFace() {
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
  }
}
function drawKeypoints() {
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
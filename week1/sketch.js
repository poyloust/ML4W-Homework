let classifier;
var dataUrl;
var cvs;
var img;
var importedImg;

// preload picture also won't work

function preload(){
  importedImg = loadImage('bird.jpg');
}

//setup canvas for drawing
function setup() {
  let c = createCanvas(windowWidth, windowHeight);
  c.id('mycanvas');
  strokeWeight(8);
  stroke(255, 204, 0);
  
}
function mouseDragged(){
  line(mouseX,mouseY,pmouseX,pmouseY);
}

// get the canvas content as base64 image with a url 
// when click the first button
//
// the url can be loaded as a background of a div element

function getCanvas(){
  cvs = document.getElementsByTagName('canvas')[0];
  document.getElementsByTagName('div')[0].style.background='url('+dataUrl+')';
  dataUrl = cvs.toDataURL();
  console.log(dataUrl);
}

function modelReady() {
  console.log('our model is ready!');
  classifier.predict(gotResult);
}
function gotResult(err, results) {
  if (results) {
    console.log('results: ', results);
    select('#result').html(results[0].className);
    select('#probability').html(results[0].probability);
    classifier.predict(gotResult);
  }
}

// load the image

function showImg(){

  image(importedImg, 0,0);


  // I can't use loadImage(dataUrl) directly here, not sure why
  // this is a method I found online that could successfully load the image from url
  
  /*  load image from canvas

  image(img,0,0,600,600);
  var raw = new Image();
  raw.src= dataUrl;
  raw.onload = function() {
    img = createImage(800, 800);
    img.drawingContext.drawImage(raw, 0, 0);
    image(img, 0, 0); 
  } 
  */

  classifier = ml5.imageClassifier('MobileNet',importedImg, modelReady);
}

function showTexts(){
  var probability = document.getElementById("probability");
  var probability_num = probability.innerText;
  if (probability_num<0.2){
    probability.style.display = 'inline-block';
  }
  else{
    probability.style.display ='none';
  }
  
}
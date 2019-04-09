let classifier;
var video;

// preload picture also won't work

function setup() {  
    noCanvas();
    video = createCapture(VIDEO);
    classifier = ml5.imageClassifier('MobileNet', video, modelReady);
  
}

function modelReady() {
  classifier.predict(gotResult);
}

function gotResult(err, results) {
    if (results) {
      //console.log('results: ', results);
      select('#result').html(results[0].className);
      select('#probability').html(results[0].probability);
    }
    showTexts();
    modelReady();
}

function showTexts(){
  var probability = document.getElementById("probability");
  var probability_num = probability.innerText;
  console.log(probability_num);
  if (probability_num<0.3){
    document.getElementById("lowPossibility").style.display = 'inline-block';
  }
  else{
    document.getElementById("lowPossibility").style.display ='none';
  }
  
}
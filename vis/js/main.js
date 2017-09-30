$(document).ready(function() {

var IMAGE_ROOT = '../demo/dataset/';
var CANVAS_HEIGHT = 400;

// Helper function to stretch a canvas
function stretchCanvas(canvas, ratio) {
  canvas.style.height = CANVAS_HEIGHT + 'px';
  canvas.style.width = Math.round(CANVAS_HEIGHT * ratio) + 'px';
  canvas.width  = canvas.offsetWidth;
  canvas.height = canvas.offsetHeight;
}

// Display results
function displayResults(data) {
  var curIndex = 0;

  var showImage = function(fname, bbox, canvasId) {
    var fpath = IMAGE_ROOT + fname;
    var img = new Image();
    img.onload = function() {  // show image and bbox
      // canvas for whole scene
      var canvas = document.getElementById(canvasId);
      var ctx = canvas.getContext('2d');
      // resize the image and canvas
      var imgRatio = img.width / img.height;
      stretchCanvas(canvas, imgRatio);
      var r = Math.min(canvas.width / img.width, canvas.height / img.height);
      var offset = Math.floor((canvas.width - Math.floor(img.width * r)) / 2);
      ctx.drawImage(img, offset, 0,
                    Math.floor(img.width*r),
                    Math.floor(img.height*r));
      // draw bounding box
      var pad = 10;
      for (var i=0;i<bbox.length;i++){
        var x = bbox[i][0], y = bbox[i][1],
            width = bbox[i][2] - bbox[i][0],
            height = bbox[i][3] - bbox[i][1];
        ctx.lineWidth = 2;
        ctx.strokeStyle = '#ff0000';
        ctx.strokeRect(offset+Math.floor(x*r), Math.floor(y*r),
                       Math.floor(width*r), Math.floor(height*r));
      }
      
    };
    img.src = fpath;
  };


  // Append or remove canvas HTML elements for gallery
  var initCanvasList = function(divId, num) {
    var div = $('#' + divId);
    while (div.children().length < num) {
      div.append('<canvas id="' + divId + '-canvas-' +
                 div.children().length + '"></canvas>')
    }
    // $('#' + divId + ' canvas:gt(' + (num-1) + ')').remove();
  };

  // Render results on the current index
  var render = function() {
    var item = data[curIndex];
    // update header status bar
    $('#cur-id-text').val(curIndex + 1);
    $('#counter').text(' / ' + data.length);

    $('#gt1').val(item.gt[0]);
    $('#gt2').val(item.gt[1]);
    $('#det1').val(item.det[0]);
    $('#det2').val(item.det[1]);
    $('#det3').val(item.det[2]);

    initCanvasList('probe', 1);
    showImage(item.im_name, item.det_bbox, 'probe-canvas-0');
  };
  render();

  // User interactions
  var prev = function() {
    if (curIndex > 0) {
      --curIndex;
    }else{
      curIndex = data.length-1;
    }

    render();
  };
  var next = function() {
    if (curIndex + 1 < data.length) {
      ++curIndex;
    }else{
      curIndex = 0;
    }
    render();
  };
  var rand = function() {
    curIndex = Math.floor(Math.random() * (data.length - 1));
    render();
  };
  var jump = function() {
    var idx = parseInt($('#cur-id-text').val());
    curIndex = Math.max(0, Math.min(data.length - 1, idx - 1));
    render();
  };

  document.onkeydown = function(e) {
    var mapping = {
      37: prev,
      39: next,
      82: rand,
    };
    if (e.keyCode in mapping) mapping[e.keyCode]();
  };

  $('#prev-btn').click(prev);
  $('#next-btn').click(next);
  $('#rand-btn').click(rand);
  $('#go-btn').click(jump);
  $('#cur-id-text').keypress(function(e) {
    if (e.which === 13) jump();
  });
};

// Load json results file
function loadResults() {
  var fileURL = 'results.json?sigh=' +
                 Math.floor(Math.random() * 100000);  // prevent caching
  $.getJSON(fileURL, function(data) {
    displayResults(data);
  });
};
loadResults();

});
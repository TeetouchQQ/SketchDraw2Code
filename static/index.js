var canvas = document.getElementById('canvas');
var ctx = canvas.getContext('2d');
console.log('run script')
var container = document.getElementById('container');
canvas.width = 1000;
canvas.height = 1000;

var mouse = {x: 0, y: 0};

canvas.addEventListener('pointermove', function(e) {
  mouse.x = e.pageX - this.offsetLeft;
  mouse.y = e.pageY - this.offsetTop;
}, false);

ctx.lineJoin = 'round';
ctx.lineCap = 'round';
ctx.strokeStyle = 'black';

canvas.addEventListener('pointerdown', function(e) {
  mouse.x = e.pageX - this.offsetLeft;
  mouse.y = e.pageY - this.offsetTop;
  ctx.beginPath();
  ctx.moveTo(mouse.x, mouse.y);
  canvas.addEventListener('pointermove', onPaint, false);
}, false);

canvas.addEventListener('pointerup', function() {
  canvas.removeEventListener('pointermove', onPaint, false);
}, false);

var onPaint = function() {
  ctx.lineTo(mouse.x, mouse.y);
  ctx.lineWidth = 5;
  ctx.stroke();
};
document.getElementById("button_reset").addEventListener("click", function() {
    const context = canvas.getContext('2d');

    context.clearRect(0, 0, canvas.width, canvas.height);
  });

document.getElementById("button_get").addEventListener("click", function UploadPic() {
	// Generate the image data
	console.log('send')
	var Pic = document.getElementById("canvas").toDataURL("image/jpg");
	Pic = Pic.replace(/^data:image\/(png|jpg);base64,/, "")
	// Sending the image data to Server
	console.log(Pic)
	$.ajax({
		type: 'POST',
		url: '/sketch2code',
		data: JSON.stringify({ image : Pic }),
		contentType: 'application/json;charset=UTF-8',
		dataType: 'json',
		success: function(msg,status, jqXHR){
			  var a = JSON.parse(jqXHR.responseText);

		}
	});
})
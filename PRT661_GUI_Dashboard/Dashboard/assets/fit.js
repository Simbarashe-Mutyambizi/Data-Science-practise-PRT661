(function () {
  var MIN_WIDTH = 1600;
  var MIN_HEIGHT = 900;
  var BASE_WIDTH = 1920;
  var BASE_HEIGHT = 1080;
  var MIN_SCALE = 0.72;
  var MAX_SCALE = 2;
  var NARROW = 1100;
  var root = document.documentElement;

  function fit() {
    var width = window.innerWidth;
    var height = window.innerHeight;
    var down = Math.min(width / MIN_WIDTH, height / MIN_HEIGHT);
    var up = Math.min(width / BASE_WIDTH, height / BASE_HEIGHT);
    var scale = 1;
    if (width <= NARROW) {
      scale = 1;
    } else if (down < 1) {
      scale = Math.max(MIN_SCALE, down);
    } else if (up > 1) {
      scale = Math.min(MAX_SCALE, up);
    }
    root.style.setProperty("--fit", scale.toFixed(4));
    root.classList.toggle("fit-scaled", scale !== 1);
  }

  fit();
  window.addEventListener("resize", fit);
})();

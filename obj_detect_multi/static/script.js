function openAct(evt, actName) {
  var i, tabcontent, tablinks;
  tabcontent = document.getElementsByClassName("tabcontent");
  for (i = 0; i < tabcontent.length; i++) {
    tabcontent[i].style.display = "none";
  }
  tablinks = document.getElementsByClassName("tablinks");
  for (i = 0; i < tablinks.length; i++) {
    tablinks[i].className = tablinks[i].className.replace(" active", "");
  }
  document.getElementById(actName).style.display = "block";
  evt.currentTarget.className += " active";
}

// Get the element with id="defaultOpen" and click on it
document.getElementById("defaultOpen").click();

function d_load() {
  document.getElementById("imgOpen").click();
}

function b_nav() {
  window.history.back();
}

function ChangePhoto(name, img) {
  img = typeof img !== 'undefined' ? img : "{{ result['original'] }}";
  target = document.getElementById("label");
  target.innerHTML = name;
  target = document.getElementById("photo");
  target.src = img;
}

function WaitDisplay(upName) {
  target = document.getElementById("result");
  target.style.display = "none";
  target = document.getElementById("loading");
  target.style.display = "";
  setTimeout(function() {
    document.getElementById(upName).submit();
  }, 100);
}

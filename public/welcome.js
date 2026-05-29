function injectContent() {
  if (document.getElementById('dgpt-custom-text')) return;

  var welcomeScreen = document.getElementById('welcome-screen');
  if (!welcomeScreen) return;

  var logo = welcomeScreen.querySelector('img.logo');
  if (!logo) return;

  var container = document.createElement('div');
  container.id = 'dgpt-custom-text';
  container.innerHTML =
    '<h2 style="text-align:center;font-size:1.5rem;font-weight:700;color:#d32f2f;' +
    'letter-spacing:0.05em;margin:0.5rem 0 0;">DictateurGPT</h2>' +
    '<p style="text-align:center;color:#999;font-size:0.875rem;max-width:400px;' +
    'margin:0.25rem auto 0;">Assistant specialisé sur les dictateurs et régimes autoritaires</p>';

  logo.insertAdjacentElement('afterend', container);
}

function removeContent() {
  var el = document.getElementById('dgpt-custom-text');
  if (el) el.remove();
}

var observer = new MutationObserver(function () {
  if (document.getElementById('welcome-screen')) {
    injectContent();
  } else {
    removeContent();
  }
});

observer.observe(document.body, { childList: true, subtree: true });

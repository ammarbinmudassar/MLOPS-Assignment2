const form = document.querySelector('[data-upload-form]');
if (form) {
  const button = form.querySelector('button[type="submit"]');
  const status = form.querySelector('[data-upload-status]');
  form.addEventListener('submit', () => {
    button.disabled = true;
    button.textContent = 'Checking image…';
    form.setAttribute('aria-busy', 'true');
    status.textContent = 'Checking the image. This may take a moment.';
  });
  window.addEventListener('pageshow', () => {
    button.disabled = false;
    button.textContent = 'Check image';
    form.removeAttribute('aria-busy');
    status.textContent = '';
  });
}

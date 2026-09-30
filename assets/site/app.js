/* Title, results carousel, improvement loop, section navigation, and the recorded demo. */
(() => {
  'use strict';

  const root = document.documentElement;
  const clamp = value => Math.max(0, Math.min(1, value));
  const header = document.querySelector('.site-header');
  const links = [...document.querySelectorAll('.site-nav a')];
  const sections = links.map(link => document.querySelector(link.getAttribute('href'))).filter(Boolean);
  const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');
  const target = document.getElementById('typed-phrase');
  const titleMarkup = target.innerHTML;
  const phrases = [
    'Advancing Self-Improving Agents\nthrough Long-Horizon Reflective Tasks.',
    'Learning to improve,\nround after round.',
    'From coding tasks\nto deep research.',
  ];
  let phraseIndex = 0;
  let characterIndex = phrases[0].length;
  let deleting = true;
  let typingTimer = 0;
  let titleVisible = true;

  const renderPhrase = () => {
    const lines = phrases[phraseIndex].slice(0, characterIndex).split('\n');
    target.replaceChildren(document.createTextNode(lines[0]));
    if (lines.length > 1) {
      const lineBreak = document.createElement('br');
      lineBreak.className = 'title-break';
      target.append(lineBreak, document.createTextNode(` ${lines[1]}`));
    }
  };

  const typeNext = () => {
    if (reducedMotion.matches) return;
    if (document.hidden || !titleVisible) {
      typingTimer = setTimeout(typeNext, 400);
      return;
    }
    const phrase = phrases[phraseIndex];
    characterIndex += deleting ? -1 : 1;
    renderPhrase();
    let delay = deleting ? 22 : 58;
    if (!deleting && characterIndex === phrase.length) {
      deleting = true;
      delay = 3400;
    } else if (deleting && characterIndex === 0) {
      deleting = false;
      phraseIndex = (phraseIndex + 1) % phrases.length;
      delay = 360;
    }
    typingTimer = setTimeout(typeNext, delay);
  };

  const resetTitle = () => {
    clearTimeout(typingTimer);
    target.innerHTML = titleMarkup;
    phraseIndex = 0;
    characterIndex = phrases[0].length;
    deleting = true;
    if (!reducedMotion.matches) typingTimer = setTimeout(typeNext, 3400);
  };
  new IntersectionObserver(entries => {
    titleVisible = entries[0].isIntersecting;
  }).observe(document.getElementById('hero-title'));
  reducedMotion.addEventListener('change', resetTitle);
  resetTitle();

  const results = document.getElementById('results');
  const resultsViewport = results.querySelector('.results-viewport');
  const resultSlides = [...results.querySelectorAll('[data-result-slide]')];
  const resultCaptions = [...results.querySelectorAll('[data-result-caption]')];
  const resultSelectors = [...results.querySelectorAll('[data-result-index]')];
  const resultPlayback = results.querySelector('.results-playback');
  let resultIndex = 0;
  let resultTimer = 0;
  let resultsVisible = false;
  let resultsHovered = false;
  let resultsFocusPaused = false;
  let resultsRotating = !reducedMotion.matches;

  const scheduleResults = () => {
    clearTimeout(resultTimer);
    resultPlayback.classList.toggle('is-paused', !resultsRotating);
    const label = resultsRotating ? 'Pause automatic figure rotation' : 'Start automatic figure rotation';
    resultPlayback.setAttribute('aria-label', label);
    resultPlayback.title = label;
    if (!resultsRotating || !resultsVisible || resultsHovered || resultsFocusPaused || document.hidden) return;
    resultTimer = setTimeout(() => {
      const next = (resultIndex + 1) % resultSlides.length;
      const nextImages = [...resultSlides[next].querySelectorAll('img')];
      // Keep the current figure visible until every image in the next slide is ready.
      if (nextImages.every(image => image.complete && image.naturalWidth)) showResult(next);
      scheduleResults();
    }, 4000);
  };
  const showResult = index => {
    resultIndex = index;
    results.dataset.activeResult = String(index);
    resultSlides.forEach((slide, i) => {
      const active = i === index;
      slide.classList.toggle('is-active', active);
      slide.setAttribute('aria-hidden', String(!active));
      slide.inert = !active;
      resultCaptions[i].classList.toggle('is-active', active);
      resultCaptions[i].setAttribute('aria-hidden', String(!active));
      resultSelectors[i].setAttribute('aria-pressed', String(active));
    });
    resultsViewport.scrollLeft = 0;
  };
  const selectResult = index => {
    resultsRotating = false;
    showResult(index);
    scheduleResults();
  };
  resultSelectors.forEach((button, index) => {
    button.addEventListener('click', () => selectResult(index));
    button.addEventListener('keydown', event => {
      const directions = {ArrowLeft: -1, ArrowRight: 1};
      if (!(event.key in directions)) return;
      event.preventDefault();
      const next = (index + directions[event.key] + resultSlides.length) % resultSlides.length;
      resultSelectors[next].focus();
      selectResult(next);
    });
  });
  resultPlayback.addEventListener('click', () => {
    resultsRotating = !resultsRotating;
    if (resultsRotating) resultsFocusPaused = false;
    scheduleResults();
  });
  results.addEventListener('pointerenter', event => {
    if (event.pointerType === 'touch') return;
    resultsHovered = true;
    scheduleResults();
  });
  results.addEventListener('pointerleave', () => {
    resultsHovered = false;
    scheduleResults();
  });
  resultsViewport.addEventListener('pointerdown', () => {
    resultsRotating = false;
    scheduleResults();
  });
  results.addEventListener('focusin', () => {
    resultsFocusPaused = true;
    scheduleResults();
  });
  results.addEventListener('focusout', () => setTimeout(() => {
    if (!results.contains(document.activeElement)) resultsFocusPaused = false;
    scheduleResults();
  }, 0));
  document.addEventListener('visibilitychange', scheduleResults);
  reducedMotion.addEventListener('change', () => {
    if (reducedMotion.matches) resultsRotating = false;
    scheduleResults();
  });
  new IntersectionObserver(entries => {
    resultsVisible = entries[0].isIntersecting && entries[0].intersectionRatio >= .25;
    scheduleResults();
  }, {threshold: [0, .25]}).observe(resultsViewport);
  results.querySelector('.results-controls').hidden = false;
  scheduleResults();

  const progressPlot = document.querySelector('.progress-illustration');
  const curve = document.getElementById('progress-curve');
  const reveal = document.getElementById('progress-reveal');
  const cursor = document.getElementById('progress-cursor');
  const dot = document.getElementById('progress-dot');
  const halo = document.getElementById('progress-halo');
  const loopSteps = [...document.querySelectorAll('[data-loop-step]')];
  const curveLength = curve.getTotalLength();
  let curveFrame = 0;
  let curveStarted = false;
  let curveVisible = false;

  const drawImprovement = progress => {
    const point = curve.getPointAtLength(curveLength * progress);
    curve.style.strokeDasharray = `${curveLength * progress} ${curveLength}`;
    reveal.setAttribute('width', String(point.x - 30));
    cursor.setAttribute('d', `M${point.x} ${point.y}V318`);
    for (const node of [dot, halo]) {
      node.setAttribute('cx', String(point.x));
      node.setAttribute('cy', String(point.y));
    }
    // Repeat the same loop as the conceptual best-so-far curve advances.
    const activeStep = progress > 0 && progress < 1 ? Math.floor(progress * 12) % 4 : -1;
    loopSteps.forEach((node, index) => node.classList.toggle('is-active', index === activeStep));
  };
  const resetImprovement = () => {
    cancelAnimationFrame(curveFrame);
    curveFrame = 0;
    curveStarted = false;
    drawImprovement(reducedMotion.matches ? 1 : 0);
  };
  const startImprovement = () => {
    if (curveStarted || reducedMotion.matches || document.hidden) return;
    curveStarted = true;
    let startedAt;
    const animate = now => {
      startedAt ??= now;
      const progress = clamp((now - startedAt) / 5000);
      drawImprovement(progress);
      curveFrame = progress < 1 ? requestAnimationFrame(animate) : 0;
    };
    curveFrame = requestAnimationFrame(animate);
  };
  resetImprovement();
  new IntersectionObserver(entries => {
    const entry = entries[0];
    curveVisible = entry.isIntersecting && entry.intersectionRatio >= .35;
    if (!entry.isIntersecting) resetImprovement();
    else if (curveVisible) startImprovement();
  }, {threshold: [0, .35]}).observe(progressPlot);
  reducedMotion.addEventListener('change', () => {
    resetImprovement();
    if (curveVisible) startImprovement();
  });
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) resetImprovement();
    else if (curveVisible) startImprovement();
  });

  const demoSection = document.getElementById('demo');
  const demoView = document.getElementById('demo-view');
  const demo = document.getElementById('run-demo');
  const demoToolbar = document.querySelector('.demo-toolbar');
  const demoViewLabel = document.getElementById('demo-view-label');
  const precedingContent = [...document.querySelectorAll('#main-content > :not(#demo), .site-footer')];
  const targetOrigin = location.origin === 'null' ? '*' : location.origin;
  let frame = 0;
  let demoOpen = false;
  let demoScrollDocument = null;

  root.classList.add('demo-view-enabled');
  demoView.inert = true;

  const syncDemoPlayback = () => {
    demo.contentWindow?.postMessage({type: demoOpen ? 'arex-demo:play' : 'arex-demo:pause'}, targetOrigin);
  };
  const sizeDemo = () => {
    const height = Math.max(580, innerHeight - demoToolbar.offsetHeight - 44);
    demo.contentWindow?.postMessage({type: 'arex-demo:viewport', height}, targetOrigin);
  };
  // While the demo slides into view, scroll the page instead of the iframe.
  const bindDemoScroll = () => {
    const frameDocument = demo.contentDocument;
    if (!frameDocument || frameDocument === demoScrollDocument) return;
    demoScrollDocument = frameDocument;
    frameDocument.addEventListener('wheel', event => {
      if (demoOpen || event.ctrlKey || !event.deltaY) return;
      event.preventDefault();
      const unit = event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? innerHeight : 1;
      scrollBy({top: event.deltaY * unit, behavior: 'instant'});
    }, {passive: false});
    let touchY = null;
    frameDocument.addEventListener('touchstart', event => {
      touchY = !demoOpen && event.touches.length === 1 ? event.touches[0].screenY : null;
    }, {passive: true});
    frameDocument.addEventListener('touchmove', event => {
      if (touchY === null || event.touches.length !== 1) { touchY = null; return; }
      const nextY = event.touches[0].screenY;
      const delta = touchY - nextY;
      touchY = nextY;
      event.preventDefault();
      (demoOpen ? demoView : window).scrollBy({top: delta, behavior: 'instant'});
    }, {passive: false});
    const endTouch = () => { touchY = null; };
    frameDocument.addEventListener('touchend', endTouch, {passive: true});
    frameDocument.addEventListener('touchcancel', endTouch, {passive: true});
  };
  const setDemoOpen = open => {
    if (open === demoOpen) return;
    const hadDemoFocus = demoView.contains(document.activeElement);
    demoOpen = open;
    root.classList.toggle('demo-is-open', open);
    header.inert = open;
    precedingContent.forEach(section => { section.inert = open; });
    if (open) {
      demoView.scrollTop = 0;
      demoView.focus({preventScroll: true});
    } else if (hadDemoFocus) {
      header.querySelector('.wordmark').focus({preventScroll: true});
    }
    syncDemoPlayback();
  };

  const update = () => {
    frame = 0;
    const headerHeight = header.offsetHeight;
    const demoOffset = Math.max(0, demoSection.getBoundingClientRect().top - headerHeight);
    const demoProgress = clamp(1 - demoOffset / Math.max(1, innerHeight));
    root.style.setProperty('--demo-offset', `${demoOffset}px`);
    root.style.setProperty('--demo-progress', String(demoProgress));
    root.style.setProperty('--page-opacity', String(1 - demoProgress));
    const wasVisible = root.classList.contains('demo-is-visible');
    const visible = demoProgress > 0;
    root.classList.toggle('demo-is-visible', visible);
    demoView.inert = !visible;
    setDemoOpen(demoOffset < 1);
    if (wasVisible && !visible) syncDemoPlayback();

    let active = demoOpen ? demoSection : undefined;
    if (!demoOpen) {
      for (const section of sections) {
        if (section.getBoundingClientRect().top <= headerHeight + 24) active = section;
      }
    }
    links.forEach(link => {
      if (active && link.getAttribute('href') === `#${active.id}`) link.setAttribute('aria-current', 'location');
      else link.removeAttribute('aria-current');
    });
  };
  const schedule = () => { if (!frame) frame = requestAnimationFrame(update); };
  window.addEventListener('scroll', schedule, {passive: true});
  window.addEventListener('resize', () => {
    sizeDemo();
    if (demoOpen) scrollTo({top: demoSection.getBoundingClientRect().top + scrollY - header.offsetHeight, behavior: 'instant'});
    schedule();
  });
  window.addEventListener('load', schedule);
  window.addEventListener('hashchange', schedule);
  demo.addEventListener('load', () => { bindDemoScroll(); sizeDemo(); syncDemoPlayback(); });
  window.addEventListener('message', event => {
    if (event.source !== demo.contentWindow || event.origin !== location.origin) return;
    if (event.data?.type === 'arex-demo:view' && ['replay','video'].includes(event.data.view)) {
      demoViewLabel.textContent = event.data.view === 'video' ? 'Demo video' : 'Interactive demo';
    }
    if (event.data?.type === 'arex-demo:size' && Number.isFinite(event.data.height)) {
      demo.style.height = `${Math.max(320, Math.min(2400, event.data.height))}px`;
      schedule();
    }
  });
  // Figures and web fonts can change section positions without a window resize.
  new ResizeObserver(schedule).observe(document.getElementById('main-content'));
  bindDemoScroll();
  sizeDemo();
  update();
})();

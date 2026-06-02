document.addEventListener('DOMContentLoaded', () => {
  // ==========================================================================
  // Sticky Header Effect
  // ==========================================================================
  const header = document.getElementById('header');
  
  const handleScroll = () => {
    if (window.scrollY > 20) {
      header.classList.add('scrolled');
    } else {
      header.classList.remove('scrolled');
    }
  };

  window.addEventListener('scroll', handleScroll);
  // Run on load in case page starts scrolled
  handleScroll();

  // ==========================================================================
  // Mobile Menu Toggle
  // ==========================================================================
  const menuToggle = document.getElementById('menu-toggle');
  const mobileMenu = document.getElementById('mobile-menu');

  if (menuToggle && mobileMenu) {
    menuToggle.addEventListener('click', () => {
      const isOpen = mobileMenu.classList.toggle('open');
      
      // Update SVG icon based on state (Hamburger vs Close)
      if (isOpen) {
        menuToggle.innerHTML = `
          <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="menu-icon">
            <line x1="18" x2="6" y1="6" y2="18"></line>
            <line x1="6" x2="18" y1="6" y2="18"></line>
          </svg>
        `;
        menuToggle.setAttribute('aria-expanded', 'true');
      } else {
        menuToggle.innerHTML = `
          <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="menu-icon">
            <line x1="4" x2="20" y1="12" y2="12"></line>
            <line x1="4" x2="20" y1="6" y2="6"></line>
            <line x1="4" x2="20" y1="18" y2="18"></line>
          </svg>
        `;
        menuToggle.setAttribute('aria-expanded', 'false');
      }
    });

    // Close mobile menu when clicking a link
    const mobileLinks = mobileMenu.querySelectorAll('a');
    mobileLinks.forEach(link => {
      link.addEventListener('click', () => {
        mobileMenu.classList.remove('open');
        menuToggle.innerHTML = `
          <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="menu-icon">
            <line x1="4" x2="20" y1="12" y2="12"></line>
            <line x1="4" x2="20" y1="6" y2="6"></line>
            <line x1="4" x2="20" y1="18" y2="18"></line>
          </svg>
        `;
        menuToggle.setAttribute('aria-expanded', 'false');
      });
    });
  }

  // ==========================================================================
  // FAQ Accordion (Single-Collapsible)
  // ==========================================================================
  const accordionHeaders = document.querySelectorAll('.accordion-header');

  accordionHeaders.forEach(header => {
    header.addEventListener('click', () => {
      const currentItem = header.parentElement;
      const currentContent = currentItem.querySelector('.accordion-content');
      const isAlreadyActive = currentItem.classList.contains('active');

      // Collapse all other items
      document.querySelectorAll('.accordion-item').forEach(item => {
        if (item !== currentItem) {
          item.classList.remove('active');
          const content = item.querySelector('.accordion-content');
          content.style.maxHeight = null;
        }
      });

      // Toggle current item
      if (isAlreadyActive) {
        currentItem.classList.remove('active');
        currentContent.style.maxHeight = null;
      } else {
        currentItem.classList.add('active');
        // Set dynamic max-height based on content height to enable smooth CSS transition
        currentContent.style.maxHeight = currentContent.scrollHeight + "px";
      }
    });
  });

  // ==========================================================================
  // Scroll Reveal Animations
  // ==========================================================================
  const revealElements = document.querySelectorAll('.reveal');

  if ('IntersectionObserver' in window) {
    const revealObserver = new IntersectionObserver((entries, observer) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          entry.target.classList.add('active');
          observer.unobserve(entry.target); // Trigger once
        }
      });
    }, {
      threshold: 0.1,
      rootMargin: '0px 0px -40px 0px'
    });

    revealElements.forEach(el => revealObserver.observe(el));
  } else {
    // Fallback if IntersectionObserver is not supported
    revealElements.forEach(el => el.classList.add('active'));
  }

});

document.addEventListener("DOMContentLoaded", () => {
  // Mobile Menu Toggle
  const menuToggle = document.querySelector(".menu-toggle")
  const navLinks = document.querySelector(".nav-links")
  const navButtons = document.querySelector(".nav-buttons")

  if (menuToggle) {
    menuToggle.addEventListener("click", () => {
      navLinks.classList.toggle("active")
      navButtons.classList.toggle("active")

      // Toggle menu icon
      const icon = menuToggle.querySelector("i")
      if (icon.classList.contains("fa-bars")) {
        icon.classList.remove("fa-bars")
        icon.classList.add("fa-times")
      } else {
        icon.classList.remove("fa-times")
        icon.classList.add("fa-bars")
      }
    })
  }

  // Smooth Scrolling for Anchor Links
  document.querySelectorAll('a[href^="#"]').forEach((anchor) => {
    anchor.addEventListener("click", function (e) {
      e.preventDefault()

      const targetId = this.getAttribute("href")
      if (targetId === "#") return

      const targetElement = document.querySelector(targetId)
      if (targetElement) {
        // Close mobile menu if open
        if (navLinks.classList.contains("active")) {
          navLinks.classList.remove("active")
          navButtons.classList.remove("active")
          const icon = menuToggle.querySelector("i")
          icon.classList.remove("fa-times")
          icon.classList.add("fa-bars")
        }

        window.scrollTo({
          top: targetElement.offsetTop - 80,
          behavior: "smooth",
        })
      }
    })
  })

  // Course Tabs
  const tabBtns = document.querySelectorAll(".tab-btn")
  const tabPanes = document.querySelectorAll(".tab-pane")

  tabBtns.forEach((btn) => {
    btn.addEventListener("click", function () {
      const tabId = this.getAttribute("data-tab")

      // Remove active class from all buttons and panes
      tabBtns.forEach((btn) => btn.classList.remove("active"))
      tabPanes.forEach((pane) => pane.classList.remove("active"))

      // Add active class to current button and pane
      this.classList.add("active")
      document.getElementById(tabId).classList.add("active")
    })
  })

  // Infrastructure Tabs
  const infraTabs = document.querySelectorAll(".infra-tab")
  const infraPanes = document.querySelectorAll(".infra-tab-pane")

  infraTabs.forEach((tab) => {
    tab.addEventListener("click", function () {
      const tabId = this.getAttribute("data-tab")

      // Remove active class from all tabs and panes
      infraTabs.forEach((tab) => tab.classList.remove("active"))
      infraPanes.forEach((pane) => pane.classList.remove("active"))

      // Add active class to current tab and pane
      this.classList.add("active")
      document.getElementById(tabId).classList.add("active")
    })
  })

  // Back to Top Button
  const backToTopBtn = document.querySelector(".back-to-top")

  window.addEventListener("scroll", () => {
    if (window.pageYOffset > 300) {
      backToTopBtn.classList.add("active")
    } else {
      backToTopBtn.classList.remove("active")
    }
  })

  // Form Submission
  const contactForm = document.getElementById("contactForm")

  if (contactForm) {
    contactForm.addEventListener("submit", function (e) {
      e.preventDefault()

      // Get form data
      const formData = new FormData(this)
      const formObject = {}

      formData.forEach((value, key) => {
        formObject[key] = value
      })

      // Here you would typically send the data to your Flask backend
      // For demonstration, we'll just show an alert
      alert("Form submitted successfully! We will contact you soon.")

      // Reset form
      this.reset()
    })
  }

  // Add mobile-specific styles
  function handleMobileStyles() {
    if (window.innerWidth <= 768) {
      // Add mobile-specific CSS
      document.body.classList.add("mobile")

      // Add mobile menu styles
      const style = document.createElement("style")
      style.id = "mobile-styles"
      style.innerHTML = `
                .nav-links.active, .nav-buttons.active {
                    display: flex;
                    flex-direction: column;
                    position: absolute;
                    top: 80px;
                    left: 0;
                    width: 100%;
                    background-color: white;
                    padding: 20px;
                    box-shadow: 0 5px 10px rgba(0, 0, 0, 0.1);
                    z-index: 999;
                }
                
                .nav-links.active {
                    align-items: center;
                }
                
                .nav-links.active li {
                    margin: 10px 0;
                }
                
                .nav-buttons.active {
                    padding-top: 0;
                    align-items: center;
                    gap: 15px;
                }
            `

      if (!document.getElementById("mobile-styles")) {
        document.head.appendChild(style)
      }
    } else {
      document.body.classList.remove("mobile")
      const mobileStyles = document.getElementById("mobile-styles")
      if (mobileStyles) {
        mobileStyles.remove()
      }
    }
  }

  // Initial call and event listener for resize
  handleMobileStyles()
  window.addEventListener("resize", handleMobileStyles)

  // Add animation on scroll
  const animateOnScroll = () => {
    const elements = document.querySelectorAll(
      ".section-header, .about-content, .course-card, .excellence-card, .achievement-card, .infra-content, .activities-content, .certification-card, .sports-content, .contact-content",
    )

    elements.forEach((element) => {
      const elementPosition = element.getBoundingClientRect().top
      const windowHeight = window.innerHeight

      if (elementPosition < windowHeight - 100) {
        element.classList.add("animate")
      }
    })
  }

  // Add animation class
  const style = document.createElement("style")
  style.innerHTML = `
        .section-header, .about-content, .course-card, .excellence-card, .achievement-card, .infra-content, .activities-content, .certification-card, .sports-content, .contact-content {
            opacity: 0;
            transform: translateY(30px);
            transition: opacity 0.6s ease, transform 0.6s ease;
        }
        
        .animate {
            opacity: 1;
            transform: translateY(0);
        }
    `
  document.head.appendChild(style)

  // Initial call and event listener for scroll
  animateOnScroll()
  window.addEventListener("scroll", animateOnScroll)
})
document.addEventListener('DOMContentLoaded', () => {
    const chatbotIcon = document.getElementById('chatbot-icon');
    const chatbotWindow = document.getElementById('chatbot-window');
    const closeChatbot = document.getElementById('close-chatbot');
    const sendButton = document.getElementById('send-message');
    const chatbotInput = document.getElementById('chatbot-input');
    const chatbotBody = document.getElementById('chatbot-body');
    const welcomeMessage = document.getElementById('welcome-message');

    // Toggle chatbot window
    chatbotIcon.addEventListener('click', () => {
        chatbotWindow.classList.toggle('hidden');
    });

    // Close chatbot window
    closeChatbot.addEventListener('click', () => {
        chatbotWindow.classList.add('hidden');
    });

    // Send message on button click
    sendButton.addEventListener('click', sendMessage);

    // Send message on Enter key press
    chatbotInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            sendMessage();
        }
    });

    // Toggle welcome message every 5 seconds
    let isVisible = false;
    function toggleWelcomeMessage() {
        welcomeMessage.style.display = isVisible ? 'none' : 'block';
        isVisible = !isVisible;
    }
    toggleWelcomeMessage(); // Show immediately on load
    setInterval(toggleWelcomeMessage, 5000);

    // Send message to Flask backend
    function sendMessage() {
        const message = chatbotInput.value.trim();
        if (!message) return;

        // Append user message
        const userMessage = document.createElement('div');
        userMessage.className = 'user-message';
        userMessage.textContent = message;
        chatbotBody.appendChild(userMessage);
        chatbotBody.scrollTop = chatbotBody.scrollHeight;

        // Clear input
        chatbotInput.value = '';

        // Send message to Flask /chat endpoint
        fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
            },
            body: `message=${encodeURIComponent(message)}`
        })
        .then(response => response.json())
        .then(data => {
            // Append bot response
            const botMessage = document.createElement('div');
            botMessage.className = 'bot-message';
            botMessage.textContent = data.success ? data.message : `Error: ${data.message}`;
            chatbotBody.appendChild(botMessage);
            chatbotBody.scrollTop = chatbotBody.scrollHeight;
        })
        .catch(error => {
            console.error('Error:', error);
            const botMessage = document.createElement('div');
            botMessage.className = 'bot-message';
            botMessage.textContent = 'Error communicating with the server.';
            chatbotBody.appendChild(botMessage);
            chatbotBody.scrollTop = chatbotBody.scrollHeight;
        });
    }
});
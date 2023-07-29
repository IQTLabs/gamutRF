document.querySelectorAll(".nav-link").forEach((link) => {
    link.addEventListener("click", function(e) {
        e.preventDefault();
        
        document.querySelectorAll(".nav-link").forEach((navLink) => {
            navLink.classList.remove("active");
        });
        e.target.classList.add("active");
        
        document.querySelectorAll(".tab-pane").forEach((pane) => {
            pane.classList.remove("show", "active");
        });
        document.querySelector(e.target.getAttribute("href")).classList.add("show", "active");
    });
});

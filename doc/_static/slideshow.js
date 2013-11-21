$(document).ready(function() 
{
    var FADE_TIME = 250;
    var IMAGE_TIME = 5000;
    var curr_index = 0;
    var images = $("#gallery img");
    images.hide()
    if (images.length > 0)
    { 
	images.eq(0).show();
    }
    var slideshow_timer = setInterval(switch_image, IMAGE_TIME);
    
    function switch_image()
    {
	old_index = curr_index;
	if (curr_index < (images.length-1)) {
	    curr_index += 1; 
	}
	else {
	    curr_index = 0;
	}
	show_hide(curr_index, old_index);
    }
    
    function show_hide(show_index, hide_index)
    {
	images.eq(hide_index).fadeOut(FADE_TIME);
	images.eq(show_index).delay(FADE_TIME+100).fadeIn(FADE_TIME);
    }
});

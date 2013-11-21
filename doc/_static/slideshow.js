$(document).ready(function() 
{
    // Constants to be redefined
    var FADE_TIME = 250;
    var IMAGE_TIME = 5000;

    var curr_index = 0;
    var elements = $("#gallery").children();
    elements.hide()
    if (elements.length > 0)
    { 
	elements.eq(0).show();
    }
    var slideshow_timer = setInterval(switch_image, IMAGE_TIME);
    
    function switch_image()
    {
	old_index = curr_index;
	if (curr_index < (elements.length-1)) {
	    curr_index += 1; 
	}
	else {
	    curr_index = 0;
	}
	show_hide(curr_index, old_index);
    }
    
    function show_hide(show_index, hide_index)
    {
	elements.eq(hide_index).fadeOut(FADE_TIME);
	elements.eq(show_index).delay(FADE_TIME+100).fadeIn(FADE_TIME);
    }
});

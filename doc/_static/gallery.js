jQuery(document).ready(function ($) {

        var _SlideshowTransitions = [
            //Fade
            { $Duration: 1200, $Opacity: 2 }
        ];

        var options = {
            $AutoPlay: true,
            $AutoPlaySteps: 1,
            $AutoPlayInterval: 3000,
            $PauseOnHover: 1,
            $ArrowKeyNavigation: true,
            $SlideDuration: 500,
            $MinDragOffsetToSlide: 20,
            $SlideSpacing: 0,
            $DisplayPieces: 1,
            $ParkingPosition: 0,
            $UISearchMode: 1,
            $PlayOrientation: 1,
            $DragOrientation: 3,

            $SlideshowOptions: {
                $Class: $JssorSlideshowRunner$,
                $Transitions: _SlideshowTransitions,
                $TransitionsOrder: 1,
                $ShowLink: false
            },

            $ArrowNavigatorOptions: {
                $Class: $JssorArrowNavigator$,
                $ChanceToShow: 1,
                $AutoCenter: 0,
                $Steps: 1
            }
        };

        var jssor_gallery = new $JssorSlider$("gallery", options);
    });
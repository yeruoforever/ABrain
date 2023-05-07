# version 330 core

layout (location = 0) in vec2 pos_screen;

out vec3 tex_coord;

uniform int plane;
uniform vec2 WH;
uniform vec3 ABC;
uniform vec3 center;

void main()
{
    vec2 rscr; // range of screen (mm)
    vec3 pos;
    vec3 bias;

    gl_Position = vec4(pos_screen, 0.0, 1.0);

    rscr = 0.5 * pos_screen * WH;
    if(plane == 1){
        rscr = rscr + center.yz;
        tex_coord = vec3(center.x, rscr.x, rscr.y)/ABC;
    }
    else if(plane == 2){
        rscr = rscr + center.xz;
        tex_coord = vec3(rscr.x, center.y, rscr.y)/ABC;
    }
    else if(plane == 3){
        rscr = rscr + center.xy;
        tex_coord = vec3(rscr.x, rscr.y, center.z)/ABC;
    }
    else{
        // error;
        tex_coord = vec3(0.5, 0.5, 0.5);
    }

}
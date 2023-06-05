# version 330 core
in vec3 tex_coord;

out vec3 color;

uniform vec2 hu_range;
uniform vec3 color_bg;
uniform vec3 color_1;
uniform vec3 color_2;
uniform sampler3D img;
uniform sampler3D seg;
uniform float mix_rate;
uniform bool has_seg;

void main()
{
    float c_img;
    float c_seg;

    if(tex_coord.x<0 || tex_coord.x>1 || tex_coord.y<0 || tex_coord.y>1 || tex_coord.z<0 || tex_coord.z>1){
        color=vec3(0);
        return;
    }
    c_img = texture(img, tex_coord).r;
    c_img = clamp(c_img, hu_range.x, hu_range.y);
    c_img = (c_img-hu_range.x)/(hu_range.y - hu_range.x);

    if(has_seg){
        c_seg = texture(seg, tex_coord).r;
        if(c_seg<0.9){
            color = vec3(c_img);
        }
        else if(0.9 < c_seg && c_seg<1.1){
            color = mix(vec3(c_img), color_1.rgb, mix_rate);
        }
        else if(2.9 < c_seg && c_seg<3.1){
            color = mix(vec3(c_img), color_2.rgb, mix_rate);
        }
        else{
            color = vec3(c_img);
        }
    }
    else{
        color = vec3(c_img);
    }

    
}

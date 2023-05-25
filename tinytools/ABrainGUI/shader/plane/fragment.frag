# version 330 core
in vec3 tex_coord;

out vec3 color;

uniform vec2 hu_range;
uniform sampler3D img;

void main()
{
    float hu_min;
    float hu_max;
    float v_red;
    hu_min = hu_range.x;
    hu_max = hu_range.y;
    v_red = texture(img,tex_coord).r;
    v_red = max(v_red,hu_min);
    v_red = min(v_red,hu_max);
    v_red = v_red/(hu_max-hu_min);
    color = vec3(v_red,v_red,v_red);
}
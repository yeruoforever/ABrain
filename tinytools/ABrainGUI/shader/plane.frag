# version 330 core
in vec3 tex_coord;

out vec3 color;

uniform vec2 hu_range;
uniform sampler3D img;

void main()
{
    float v_red;
    v_red = texture(img, tex_coord).r;
    v_red = clamp(v_red, hu_range.x, hu_range.y);
    v_red = v_red/(hu_range.y - hu_range.x);
    color = vec3(v_red, v_red, v_red);
}

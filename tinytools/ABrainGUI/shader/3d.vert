# version 330 core


layout(location=0) in vec2 ndc;

out vec3 light_dir;

uniform float fov;               // 视野大小 FOV
uniform vec2 screen;             // 屏幕物理范围 W, H
uniform vec3 eye;                // 视点位置 x, y, z           
uniform mat4 V2W;                // 视野坐标转世界坐标


void main(){
    gl_Position = vec4(ndc,1.0,1.0);
    float aspect = screen.x/screen.y;
    vec4 light = -V2W*vec4(aspect*screen.x, screen.y, 1/tan(fov/2),1.0);
    light_dir = normalize(light.xyz);
}
# version 330 core


layout(location=0) in vec2 ndc;

out vec3 light_dir;

uniform vec2 screen_size;       // W, H
uniform vec3 screen_plane;      // FOV, near, far
uniform vec3 eye;               // x, y, z
//uniform mat4 W2V;             
uniform mat4 V2W;


void main(){
    gl_Position = vec4(ndc,1.0,1.0);
    float Y = tan(screen_plane.x/2)*screen_plane.y;
    float aspect = screen_size.x/screen_size.y;
    float X = aspect*Y;
    vec4 light = V2W*vec4(X*ndc.x, Y*ndc.y, -screen_plane.y ,1);
    light_dir = normalize(light.xyz);
}
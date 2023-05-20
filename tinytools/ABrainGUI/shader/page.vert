# version 330 core

layout(location=0) in vec2 ndc;

out vec2 tex_coord;

void main(){
    gl_Position = vec4(ndc.x,ndc.y,0,1);
    tex_coord = 0.5*ndc+0.5;
}
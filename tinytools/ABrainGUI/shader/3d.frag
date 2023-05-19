# version 330 core


in vec3 light_dir;

uniform vec3 eye;
uniform float step;
uniform float alpha;
uniform float vox_min;
uniform float vox_max;
uniform vec3 cube_a;
uniform vec3 cube_b;
uniform vec3 color_bg;
uniform vec3 color_1;
uniform vec3 color_2;
uniform mat4 W2M;
uniform sampler3D img;
uniform sampler3D seg;
uniform float mix_rate;

out vec4 color;


const float EPSILON = 0.000001f;

bool is_intersect(vec3 origin, vec3 light, out vec2 t){
    
    vec3 a = (cube_a-origin)/light;
    vec3 b = (cube_b-origin)/light;

    float t_min = max(max(min(a.x,b.x),min(a.y,b.y)),min(a.z,b.z));
    float t_max = min(min(max(a.x,b.x),max(a.y,b.y)),max(a.z,b.z));

    t = vec2(t_min,t_max);

    return t_min>=0 && t_min<=t_max;
}

void main(){
    // color
    vec4 dest;
    vec4 src;

    // points
    vec3 pos;

    // texture coord
    vec3 tex_coord;
    vec3 tex_bound_0=vec3(0.0);
    vec3 tex_bound_1=vec3(1.0);

    // vox value
    float vox_window=vox_max-vox_min;
    float vox_val;
    float vox_seg;
    vec3 c_mix;

    // light 
    vec2 t_range=vec2(0);
    
    dest = vec4(0);
    if(is_intersect(eye,light_dir,t_range)){
        // color = vec4((t_range.y-t_range.x)/5);
        
        for(float t=t_range.x+EPSILON;t<t_range.y;t+=step){
            pos = eye+light_dir*t;
            tex_coord = (W2M*vec4(pos,1.0)).xyz;
            vox_val = texture(img,tex_coord).r;
            vox_val = clamp(vox_val,vox_min,vox_max);
            vox_val = (vox_val-vox_min)/vox_window;
            vox_seg = texture(seg, tex_coord).r;
            if(vox_seg<0.5){
                src = vec4(vox_val);
            }
            else if (0.5 < vox_seg && vox_seg < 2.5){
                src = vec4(vox_val);
            }
            else if(2.5 < vox_seg && vox_seg < 3.5){
                src = vec4(color_1.rgb,1);
            }
            else{
                src = vec4(vox_val);
            }
            
            src.a *= alpha;
            src.rgb *= src.a;
            dest += (1-src.a)*src;
            if(dest.a > 0.99)
                break;
        }
        color = dest;
        // pos = eye+light_dir*t_range.x;
        // tex_coord = (W2M*vec4(pos,1.0)).xyz;
        // color = vec4(tex_coord,1);
    }
    else
        color = dest;
    
}
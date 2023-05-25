# version 330 core

layout (location = 0) in vec2 pos_screen;

out vec3 tex_coord;

uniform int plane;      //当前激活切片
uniform vec2 screen;    //屏幕物理范围      (vec2(0), vec2(W,H))
uniform vec3 focus;     //切片中心          (X,Y,Z)
uniform vec3 slice;     //切片平面位置      (X,Y,Z)    
uniform vec3 range;     //纹理物理范围      （vec3(0), vec3(X,Y,Z)）

void main()
{
    // texture 方向为 LPS或RAS

    vec2 tmp;
    float s;
    gl_Position = vec4(pos_screen, 0.0, 1.0);
    if(plane == 0){         // 横断面(:,:,z)
        tmp = (2*focus.xy + screen*pos_screen)/(2*range.xy)+0.5;
        tex_coord = vec3(tmp.x, tmp.y, slice.z/range.z+0.5);
    }
    else if(plane == 1){    // 冠状面（：,y,:）
        tmp = (2*focus.xz + screen*pos_screen)/(2*range.xz)+0.5;
        tex_coord = vec3(tmp.x, slice.y/range.y+0.5, tmp.y);
    }
    else if(plane == 2){    // 矢状面(x,:,:)
        tmp = (2*focus.yz + screen*pos_screen)/(2*range.yz)+0.5;
        tex_coord = vec3(slice.x/range.x+0.5, tmp.x, tmp.y);
    }
    else{                   //未定义
        // error;
        tex_coord = vec3(0.5, 0.5, 0.5);
    }

}
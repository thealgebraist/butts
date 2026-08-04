#pragma once
#include <string>
#include <vector>
#include <iostream>
#include <CoreGraphics/CoreGraphics.h>
#include <ImageIO/ImageIO.h>
#include <CoreFoundation/CoreFoundation.h>
#include <OpenGL/gl.h>

struct LoadedImage {
    GLuint textureID = 0;
    int width = 0;
    int height = 0;
};

class ImageLoader {
public:
    static LoadedImage loadTexture(const std::string& path) {
        LoadedImage img;
        CFStringRef pathString = CFStringCreateWithCString(NULL, path.c_str(), kCFStringEncodingUTF8);
        CFURLRef url = CFURLCreateWithFileSystemPath(NULL, pathString, kCFURLPOSIXPathStyle, false);
        CGImageSourceRef source = CGImageSourceCreateWithURL(url, NULL);
        
        if (!source) {
            CFRelease(url);
            CFRelease(pathString);
            return img;
        }

        CGImageRef cgImage = CGImageSourceCreateImageAtIndex(source, 0, NULL);
        if (!cgImage) {
            CFRelease(source);
            CFRelease(url);
            CFRelease(pathString);
            return img;
        }

        img.width = CGImageGetWidth(cgImage);
        img.height = CGImageGetHeight(cgImage);

        CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
        std::vector<uint8_t> data(img.width * img.height * 4);
        CGContextRef context = CGBitmapContextCreate(data.data(), img.width, img.height, 8, 4 * img.width, colorSpace, kCGImageAlphaPremultipliedLast | kCGBitmapByteOrder32Big);
        
        // Handle potential EXIF orientation
        CFDictionaryRef properties = CGImageSourceCopyPropertiesAtIndex(source, 0, NULL);
        int orientation = 1;
        if (properties) {
            CFNumberRef orientationNum = (CFNumberRef)CFDictionaryGetValue(properties, kCGImagePropertyOrientation);
            if (orientationNum) {
                CFNumberGetValue(orientationNum, kCFNumberIntType, &orientation);
            }
            CFRelease(properties);
        }

        // We draw the image
        CGContextDrawImage(context, CGRectMake(0, 0, img.width, img.height), cgImage);
        
        glGenTextures(1, &img.textureID);
        glBindTexture(GL_TEXTURE_2D, img.textureID);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, img.width, img.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, data.data());
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        CGContextRelease(context);
        CGColorSpaceRelease(colorSpace);
        CGImageRelease(cgImage);
        CFRelease(source);
        CFRelease(url);
        CFRelease(pathString);

        return img;
    }
};

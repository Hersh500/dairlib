#include "ImageToLcmImageArray.h"

#include <stdexcept>

#include <zlib.h>
#include <drake/common/value.h>

#include "dairlib/lcmt_image.hpp"
#include "dairlib/lcmt_image_array.hpp"
#include "drake/systems/sensors/lcm_image_traits.h"

using std::string;
using drake::systems::sensors::PixelType;
using drake::systems::sensors::Image;
using drake::systems::InputPort;
using drake::systems::OutputPort;

namespace dairlib {
namespace systems {
const int64_t kSecToMillisec = 1000000;


// Overwrites the msg's compression_method, size, and data.
template <drake::systems::sensors::PixelType kPixelType>
void Pack(const drake::systems::sensors::Image<kPixelType>& image, lcmt_image* msg) {
    msg->compression_method = lcmt_image::COMPRESSION_METHOD_NOT_COMPRESSED;

    const int size = image.width() * image.height() * image.kPixelSize;
    msg->data.resize(size);
    msg->size = size;
    memcpy(&msg->data[0], image.at(0, 0), size);
}

// Overwrites everything in msg except its header.
template <drake::systems::sensors::PixelType kPixelType>
void PackImageToLcmImageT(const drake::systems::sensors::Image<kPixelType>& image, lcmt_image* msg) {
    msg->width = image.width();
    msg->height = image.height();
    msg->row_stride = image.kPixelSize * msg->width;
    msg->bigendian = false;
    msg->pixel_format =
            drake::systems::sensors::LcmPixelTraits<drake::systems::sensors::ImageTraits<kPixelType>::kPixelFormat>::kPixelFormat;
    msg->channel_type = drake::systems::sensors::LcmImageTraits<kPixelType>::kChannelType;

    Pack(image, msg);
}

// Overwrites everything in msg except its header.
void PackImageToLcmImageT(const drake::AbstractValue& untyped_image,
                          drake::systems::sensors::PixelType pixel_type, lcmt_image* msg) {
    switch (pixel_type) {
        case drake::systems::sensors::PixelType::kRgb8U: {
            const auto& image_value =
                    untyped_image.get_value<drake::systems::sensors::Image<PixelType::kRgb8U>>();
            PackImageToLcmImageT(image_value, msg);
            break;
        }
        case drake::systems::sensors::PixelType::kBgr8U: {
            const auto& image_value =
                    untyped_image.get_value<drake::systems::sensors::Image<PixelType::kBgr8U>>();
            PackImageToLcmImageT(image_value, msg);
            break;
        }
        case drake::systems::sensors::PixelType::kRgba8U: {
            const auto& image_value =
                    untyped_image.get_value<drake::systems::sensors::Image<PixelType::kRgba8U>>();
            PackImageToLcmImageT(image_value, msg);
            break;
        }
        case drake::systems::sensors::PixelType::kBgra8U: {
            const auto& image_value =
                    untyped_image.get_value<drake::systems::sensors::Image<PixelType::kBgra8U>>();
            PackImageToLcmImageT(image_value, msg);
            break;
        }
        case drake::systems::sensors::PixelType::kGrey8U: {
            const auto& image_value =
                    untyped_image.get_value<drake::systems::sensors::Image<PixelType::kGrey8U>>();
            PackImageToLcmImageT(image_value, msg);
            break;
        }
        case drake::systems::sensors::PixelType::kDepth16U: {
            const auto& image_value =
                    untyped_image.get_value<drake::systems::sensors::Image<PixelType::kDepth16U>>();
            PackImageToLcmImageT(image_value, msg);
            break;
        }
        case drake::systems::sensors::PixelType::kDepth32F: {
            const auto& image_value =
                    untyped_image.get_value<Image<PixelType::kDepth32F>>();
            PackImageToLcmImageT(image_value, msg);
            break;
        }
        case drake::systems::sensors::PixelType::kLabel16I: {
            const auto& image_value =
                    untyped_image.get_value<Image<PixelType::kLabel16I>>();
            PackImageToLcmImageT(image_value, msg);
            break;
        }
        case drake::systems::sensors::PixelType::kExpr:
            throw std::domain_error("PixelType::kExpr is not supported.");
    }
}


ImageToLcmImageArrayT::ImageToLcmImageArrayT() {
    image_array_t_msg_output_port_index_ = DeclareAbstractOutputPort(
            drake::systems::kUseDefaultName, &ImageToLcmImageArrayT::CalcImageArray)
            .get_index();
}

ImageToLcmImageArrayT::ImageToLcmImageArrayT(const string& color_frame_name,
                                             const string& depth_frame_name,
                                             const string& label_frame_name) {
    color_image_input_port_index_ =
            DeclareImageInputPort<PixelType::kRgba8U>(color_frame_name).get_index();
    depth_image_input_port_index_ =
            DeclareImageInputPort<PixelType::kDepth32F>(depth_frame_name).get_index();
    label_image_input_port_index_ =
            DeclareImageInputPort<PixelType::kLabel16I>(label_frame_name).get_index();

    image_array_t_msg_output_port_index_ = DeclareAbstractOutputPort(
            drake::systems::kUseDefaultName, &ImageToLcmImageArrayT::CalcImageArray)
            .get_index();
}

const InputPort<double>& ImageToLcmImageArrayT::color_image_input_port() const {
    DRAKE_DEMAND(color_image_input_port_index_ >= 0);
    return this->get_input_port(color_image_input_port_index_);
}

const InputPort<double>& ImageToLcmImageArrayT::depth_image_input_port() const {
    DRAKE_DEMAND(depth_image_input_port_index_ >= 0);
    return this->get_input_port(depth_image_input_port_index_);
}

const InputPort<double>& ImageToLcmImageArrayT::label_image_input_port() const {
    DRAKE_DEMAND(label_image_input_port_index_ >= 0);
    return this->get_input_port(label_image_input_port_index_);
}

const OutputPort<double>& ImageToLcmImageArrayT::image_array_t_msg_output_port() const {
    return System<double>::get_output_port(image_array_t_msg_output_port_index_);
}

void ImageToLcmImageArrayT::CalcImageArray(
        const drake::systems::Context<double>& context, lcmt_image_array* msg) const {
    // A best practice for filling in LCM messages is to first value-initialize
    // the entire message to its defaults ("*msg = {}") before setting any new
    // values.  That way, if we happen to skip over any fields, they will be
    // zeroed out instead of leaving behind garbage from whatever the msg memory
    // happened to contain beforehand.
    //
    // In our case though, image data is typically high-bandwidth, so we will
    // carefully work to reuse our message vectors' storage instead of clearing
    // it on every call.  (The headers are small, though, so we'll still clear
    // them.)
    const int64_t utime = static_cast<int64_t>(
            context.get_time() * kSecToMillisec);
    msg->header = {};
    msg->header.utime = utime;
    const int num_inputs = num_input_ports();
    msg->num_images = num_inputs;
    msg->images.resize(num_inputs);
    for (int i = 0; i < num_inputs; i++) {
        const std::string& name = this->get_input_port(i).get_name();
        const PixelType& type = input_port_pixel_type_[i];
        const auto& value = this->get_input_port(i).
                template Eval<drake::AbstractValue>(context);
        lcmt_image& packed = msg->images.at(i);
        packed.header = {};
        packed.header.utime = utime;
        packed.header.frame_name = name;
        PackImageToLcmImageT(value, type, &packed);
    }
}

    }  // namespace systems
}  // namespace dairlib

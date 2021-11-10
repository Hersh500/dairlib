#include <string>
#include <vector>

#include "drake/systems/framework/leaf_system.h"
#include "dairlib/lcmt_image_array.hpp"
#include "dairlib/lcmt_image.hpp"
#include "drake/systems/sensors/image.h"
#include "drake/systems/sensors/pixel_types.h"

namespace dairlib {
namespace systems {

class ImageToLcmImageArrayT : public drake::systems::LeafSystem<double> {
public:

    /// Constructs an empty system with no input ports.
    /// After construction, use DeclareImageInputPort() to add inputs.
    explicit ImageToLcmImageArrayT();

    /// An %ImageToLcmImageArrayT constructor.  Declares three input ports --
    /// one color image, one depth image, and one label image.
    ///
    /// @param color_frame_name The frame name used for color image.
    /// @param depth_frame_name The frame name used for depth image.
    /// @param label_frame_name The frame name used for label image.
    ImageToLcmImageArrayT(const std::string& color_frame_name,
                          const std::string& depth_frame_name,
                          const std::string& label_frame_name);

    /// Returns the input port containing a color image.
    /// Note: Only valid if the color/depth/label constructor is used.
    const drake::systems::InputPort<double>& color_image_input_port() const;

    /// Returns the input port containing a depth image.
    /// Note: Only valid if the color/depth/label constructor is used.
    const drake::systems::InputPort<double>& depth_image_input_port() const;

    /// Returns the input port containing a label image.
    /// Note: Only valid if the color/depth/label constructor is used.
    const drake::systems::InputPort<double>& label_image_input_port() const;

    /// Returns the abstract valued output port that contains a
    /// `Value<lcmt_image_array>`.
    const drake::systems::OutputPort<double>& image_array_t_msg_output_port() const;

    template <drake::systems::sensors::PixelType kPixelType>
    const drake::systems::InputPort<double>& DeclareImageInputPort(const std::string& name) {
        input_port_pixel_type_.push_back(kPixelType);
        return this->DeclareAbstractInputPort(
                name, drake::Value<drake::systems::sensors::Image<kPixelType>>());
    }

private:
    void CalcImageArray(const drake::systems::Context<double>& context,
                        lcmt_image_array* msg) const;

    int color_image_input_port_index_{-1};
    int depth_image_input_port_index_{-1};
    int label_image_input_port_index_{-1};
    int image_array_t_msg_output_port_index_{-1};

    std::vector<drake::systems::sensors::PixelType> input_port_pixel_type_{};
};

}
}

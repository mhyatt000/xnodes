#!/usr/bin/env python
from __future__ import annotations

import rosbag
import rospy
from std_msgs.msg import String


def write_to_bag():
    # Open a bag file in write mode
    bag = rosbag.Bag("example.bag", "w")
    try:
        # Create a simple String message
        msg = String(data="Hello, ROS bag!")
        # Write the message to the 'chatter' topic with the current time
        bag.write("chatter", msg, rospy.Time.now())
        rospy.loginfo("Message written to bag")
    finally:
        # Always close the bag to ensure data is flushed to disk
        bag.close()


if __name__ == "__main__":
    # Initialize the ROS node
    rospy.init_node("rosbag_writer", anonymous=True)
    write_to_bag()

def get_log_data(lcm_log, lcm_channels, data_processing_callback, *args,
                 **kwargs):
    """
    Parses an LCM log and returns data as specified by a callback function
    :param lcm_log: an lcm.EventLog object
    :param lcm_channels: dictionary with entries {channel : lcmtype} of channels
    to be read from the log
    :param data_processing_callback: function pointer which takes as arguments
     (data, args, kwargs) where data is a dictionary with
     entries {CHANNEL : [ msg for lcm msg in log with msg.channel == CHANNEL ] }
    :param args: positional arguments for data_processing_callback
    :param kwargs: keyword arguments for data_processing_callback
    :return: return args of data_processing_callback
    """

    data_to_process = {}
    for event in lcm_log:
        if event.channel in lcm_channels:
            if event.channel in data_to_process:
                data_to_process[event.channel].append(
                    lcm_channels[event.channel].decode(event.data))
            else:
                data_to_process[event.channel] = \
                    [lcm_channels[event.channel].decode(event.data)]

    return data_processing_callback(data_to_process, *args, *kwargs)


def get_log_summary(lcm_log):
    channel_names_and_msg_counts = {}
    for event in lcm_log:
        if event.channel not in channel_names_and_msg_counts:
            channel_names_and_msg_counts[event.channel] = 1
        else:
            channel_names_and_msg_counts[event.channel] = \
            channel_names_and_msg_counts[event.channel] + 1
    return channel_names_and_msg_counts


def print_log_summary(filename, log):
    print(f"Channels in {filename}:\n")
    summary = get_log_summary(log)
    for channel, count in summary.items():
        print(f"{channel}: {count:06} messages")


def passthrough_callback(data, *args, **kwargs):
    return data


def main():
    import lcm
    import sys

    logfile = sys.argv[1]
    log = lcm.EventLog(logfile, "r")
    print_log_summary(logfile, log)


if __name__ == "__main__":
    main()

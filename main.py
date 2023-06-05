import utils
import scenario

def main():
    args = utils.get_args()
    utils.print_args(args)
    method = getattr(scenario, args.scenario)
    try:
        method(args)
    except KeyboardInterrupt:
        pass
        
if __name__ == '__main__':
    main()
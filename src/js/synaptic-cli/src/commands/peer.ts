import { Command } from 'commander';

export function peerCommand(): Command {
  const command = new Command('peer');
  command.description('peer command');
  return command;
}

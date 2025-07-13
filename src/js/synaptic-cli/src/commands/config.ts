import { Command } from 'commander';

export function configCommand(): Command {
  const command = new Command('config');
  command.description('config command');
  return command;
}

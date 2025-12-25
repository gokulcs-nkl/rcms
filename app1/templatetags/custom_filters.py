from django import template

register = template.Library()

@register.filter
def multiply(value, arg):
    """Multiply the value by the argument."""
    try:
        return float(value) * float(arg)
    except (ValueError, TypeError):
        return 0

@register.filter
def replace(value, args):
    """Replace old string with new string in value."""
    try:
        old, new = args.split(':')
        return value.replace(old, new)
    except (ValueError, AttributeError):
        return value

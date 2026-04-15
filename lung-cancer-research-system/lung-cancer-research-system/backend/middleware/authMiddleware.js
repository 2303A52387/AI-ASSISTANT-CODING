const ensureAuthenticated = (req, res, next) => {
  if (req.session && req.session.user) {
    return next();
  }
  req.flash('error_msg', 'Please log in to access the research dashboard');
  res.redirect('/login');
};

module.exports = { ensureAuthenticated };

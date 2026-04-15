const express = require('express');
const router = express.Router();
const bcrypt = require('bcryptjs');

let User;
try { User = require('../models/User'); } catch(e) {}

// Home redirect
router.get('/', (req, res) => {
  if (req.session && req.session.user) return res.redirect('/dashboard');
  res.redirect('/login');
});

// Login page
router.get('/login', (req, res) => {
  if (req.session && req.session.user) return res.redirect('/dashboard');
  res.render('login', { title: 'Login - Lung Cancer Research System' });
});

// Login POST
router.post('/login', async (req, res) => {
  const { email, password } = req.body;
  try {
    // Demo login fallback
    if (email === 'demo@research.edu' && password === 'demo123') {
      req.session.user = { _id: 'demo', name: 'Dr. Demo Researcher', email, institution: 'University of Research', role: 'Senior Researcher' };
      return res.redirect('/dashboard');
    }
    if (User) {
      const user = await User.findOne({ email });
      if (!user) { req.flash('error_msg', 'Invalid email or password'); return res.redirect('/login'); }
      const isMatch = await user.comparePassword(password);
      if (!isMatch) { req.flash('error_msg', 'Invalid email or password'); return res.redirect('/login'); }
      req.session.user = { _id: user._id, name: user.name, email: user.email, institution: user.institution, role: user.role };
      return res.redirect('/dashboard');
    }
    req.flash('error_msg', 'Invalid credentials. Use demo@research.edu / demo123');
    res.redirect('/login');
  } catch (err) {
    req.flash('error_msg', 'Server error. Try demo@research.edu / demo123');
    res.redirect('/login');
  }
});

// Register page
router.get('/register', (req, res) => {
  if (req.session && req.session.user) return res.redirect('/dashboard');
  res.render('register', { title: 'Register - Lung Cancer Research System' });
});

// Register POST
router.post('/register', async (req, res) => {
  const { name, email, password, password2, institution } = req.body;
  if (password !== password2) { req.flash('error_msg', 'Passwords do not match'); return res.redirect('/register'); }
  if (password.length < 6) { req.flash('error_msg', 'Password must be at least 6 characters'); return res.redirect('/register'); }
  try {
    if (User) {
      const existing = await User.findOne({ email });
      if (existing) { req.flash('error_msg', 'Email already registered'); return res.redirect('/register'); }
      const user = await User.create({ name, email, password, institution: institution || 'Research Institution' });
      req.flash('success_msg', 'Registration successful! Please log in.');
      return res.redirect('/login');
    }
    req.flash('success_msg', 'Registration simulated (no DB). Use demo@research.edu / demo123');
    res.redirect('/login');
  } catch (err) {
    req.flash('error_msg', 'Registration failed: ' + err.message);
    res.redirect('/register');
  }
});

// Logout
router.get('/logout', (req, res) => {
  req.session.destroy();
  res.redirect('/login');
});

module.exports = router;

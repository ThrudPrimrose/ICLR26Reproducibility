subroutine argmax_with_index_fp64(a, out_index, out_value, len_1d, workspace, workspace_size) bind(C)
  use iso_c_binding
  use omp_lib
  implicit none
  integer(c_int64_t), value, intent(in) :: len_1d, workspace_size
  real(c_double), intent(in) :: a(len_1d)
  integer(c_int64_t), intent(out) :: out_index(1)
  real(c_double), intent(out) :: out_value(1)
  type(c_ptr), intent(in) :: workspace

  integer :: n, i, t, nt, best
  integer(c_int64_t) :: nn, lo, hi, l8, h8
  real(c_double) :: part(64), m, bm, m0, m1, m2, m3, m4, m5, m6, m7
  integer :: firsti(64), fi, f0, f1, f2, f3, f4, f5, f6, f7

  nn = len_1d
  n = int(nn)
  if (n <= 1) then
    if (n == 1) then
      out_value(1) = a(1)
      out_index(1) = 0
    else
      out_value(1) = 0.0d0
      out_index(1) = 0
    end if
    return
  end if

  nt = 1
  !$omp parallel private(t, lo, hi, l8, h8, i, m, m0, m1, m2, m3, m4, m5, m6, m7, fi, f0, f1, f2, f3, f4, f5, f6, f7)
    t = omp_get_thread_num()
    nt = omp_get_num_threads()
    lo = (nn*t)/nt + 1
    hi = (nn*(t+1))/nt
    m = -huge(0.0d0)
    fi = n
    l8 = lo + mod(8_8 - mod(lo - 1_8, 8_8), 8_8)
    do i = int(lo), min(int(l8) - 1, int(hi))
      if (a(i) > m) then
        m = a(i)
        fi = i
      end if
    end do
    if (hi - 7_8 >= l8) then
      h8 = l8 + ((hi - l8 - 7_8) / 8_8) * 8_8
    else
      h8 = l8 - 8_8
    end if
    m0 = m; m1 = m0; m2 = m0; m3 = m0
    m4 = m0; m5 = m0; m6 = m0; m7 = m0
    f0 = fi; f1 = fi; f2 = fi; f3 = fi
    f4 = fi; f5 = fi; f6 = fi; f7 = fi
    do i = int(l8), int(h8), 8
      if (a(i)   > m0) then; m0 = a(i);   f0 = i;   end if
      if (a(i+1) > m1) then; m1 = a(i+1); f1 = i+1; end if
      if (a(i+2) > m2) then; m2 = a(i+2); f2 = i+2; end if
      if (a(i+3) > m3) then; m3 = a(i+3); f3 = i+3; end if
      if (a(i+4) > m4) then; m4 = a(i+4); f4 = i+4; end if
      if (a(i+5) > m5) then; m5 = a(i+5); f5 = i+5; end if
      if (a(i+6) > m6) then; m6 = a(i+6); f6 = i+6; end if
      if (a(i+7) > m7) then; m7 = a(i+7); f7 = i+7; end if
    end do
    m = m0
    fi = f0
    if (m1 > m) then
      m = m1; fi = f1
    else if (m1 == m .and. f1 < fi) then
      fi = f1
    end if
    if (m2 > m) then
      m = m2; fi = f2
    else if (m2 == m .and. f2 < fi) then
      fi = f2
    end if
    if (m3 > m) then
      m = m3; fi = f3
    else if (m3 == m .and. f3 < fi) then
      fi = f3
    end if
    if (m4 > m) then
      m = m4; fi = f4
    else if (m4 == m .and. f4 < fi) then
      fi = f4
    end if
    if (m5 > m) then
      m = m5; fi = f5
    else if (m5 == m .and. f5 < fi) then
      fi = f5
    end if
    if (m6 > m) then
      m = m6; fi = f6
    else if (m6 == m .and. f6 < fi) then
      fi = f6
    end if
    if (m7 > m) then
      m = m7; fi = f7
    else if (m7 == m .and. f7 < fi) then
      fi = f7
    end if
    do i = int(h8) + 8, int(hi)
      if (a(i) > m) then
        m = a(i)
        fi = i
      end if
    end do
    part(t+1) = m
    firsti(t+1) = fi
  !$omp end parallel

  bm = part(1)
  best = firsti(1)
  do t = 2, nt
    if (part(t) > bm) then
      bm = part(t)
      best = firsti(t)
    else if (part(t) == bm .and. firsti(t) < best) then
      best = firsti(t)
    end if
  end do

  out_value(1) = bm
  out_index(1) = int(best - 1, c_int64_t)
end subroutine argmax_with_index_fp64
